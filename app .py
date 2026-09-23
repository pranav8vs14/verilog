import os
import json
import subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. RTL WORKSPACE
# ============================================================

WORKDIR = Path("rtl_workspace")
WORKDIR.mkdir(exist_ok=True)


# ============================================================
# 2. VERILOG / SYSTEMVERILOG TOOLS
# ============================================================

@tool
def write_verilog_file(filename: str, content: str) -> str:
    """
    Create a Verilog/SystemVerilog source file.
    Allowed files: .v, .sv, .vh, .svh
    """
    if not filename.endswith((".v", ".sv", ".vh", ".svh")):
        return "ERROR: Only Verilog/SystemVerilog files are allowed."

    safe_name = Path(filename).name
    path = WORKDIR / safe_name

    path.write_text(content, encoding="utf-8")

    return f"Successfully created {path}"


@tool
def read_verilog_file(filename: str) -> str:
    """Read a Verilog/SystemVerilog file from the RTL workspace."""

    safe_name = Path(filename).name
    path = WORKDIR / safe_name

    if not path.exists():
        return f"ERROR: File {safe_name} does not exist."

    return path.read_text(encoding="utf-8")


@tool
def list_rtl_files() -> str:
    """List files currently present in the RTL workspace."""

    files = sorted(p.name for p in WORKDIR.iterdir())

    if not files:
        return "RTL workspace is empty."

    return "\n".join(files)


@tool
def compile_verilog(rtl_file: str, testbench_file: str) -> str:
    """
    Compile Verilog/SystemVerilog RTL and its testbench using Icarus Verilog.
    """

    rtl_path = WORKDIR / Path(rtl_file).name
    tb_path = WORKDIR / Path(testbench_file).name
    output_path = WORKDIR / "simulation.out"

    if not rtl_path.exists():
        return f"ERROR: RTL file {rtl_file} does not exist."

    if not tb_path.exists():
        return f"ERROR: Testbench file {testbench_file} does not exist."

    try:
        result = subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(output_path),
                str(rtl_path),
                str(tb_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    except FileNotFoundError:
        return (
            "ERROR: Icarus Verilog is not installed. "
            "Install iverilog and make sure it is available in PATH."
        )

    except subprocess.TimeoutExpired:
        return "ERROR: Verilog compilation timed out."

    return (
        f"EXIT_CODE={result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@tool
def run_simulation() -> str:
    """Run the compiled Verilog/SystemVerilog simulation."""

    simulation = WORKDIR / "simulation.out"

    if not simulation.exists():
        return "ERROR: No compiled simulation found. Compile the design first."

    try:
        result = subprocess.run(
            ["vvp", str(simulation)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    except FileNotFoundError:
        return "ERROR: vvp/Icarus Verilog is not installed."

    except subprocess.TimeoutExpired:
        return "ERROR: Simulation timed out."

    return (
        f"EXIT_CODE={result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


# ============================================================
# 3. TOOLS AVAILABLE TO THE AI AGENT
# ============================================================

tools = [
    write_verilog_file,
    read_verilog_file,
    list_rtl_files,
    compile_verilog,
    run_simulation,
]


# ============================================================
# 4. GEMINI MODEL
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable is not set.")


llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 5. VERILOG ENGINEERING AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt="""
You are an expert Verilog/SystemVerilog RTL design and verification engineer.

Your job is to convert a user's hardware requirement or Python algorithm
into synthesizable SystemVerilog and a self-checking testbench.

IMPORTANT HARDWARE RULES:

1. Understand the intended behavior before writing RTL.
2. Do NOT blindly translate software-only Python features into hardware.
3. Convert the algorithm into an appropriate hardware architecture.
4. Prefer SystemVerilog (.sv).
5. Clearly define inputs, outputs, bit widths, clock and reset behavior.
6. Use always_ff for sequential logic where appropriate.
7. Use always_comb for combinational logic where appropriate.
8. Use nonblocking <= assignments in sequential logic.
9. Avoid accidental latches.
10. Avoid multiple drivers for the same signal.
11. Handle signed/unsigned values and bit widths carefully.
12. Generate a self-checking testbench.
13. Include normal cases and important corner cases.
14. Use assertions where useful.
15. Create:
       design.sv
       tb.sv

VERIFICATION WORKFLOW:

Step 1:
Understand the user's specification.

Step 2:
Explain the intended hardware architecture internally.

Step 3:
Generate design.sv.

Step 4:
Generate tb.sv.

Step 5:
Compile the RTL and testbench using compile_verilog.

Step 6:
If compilation fails:
- Read the relevant source.
- Analyze the compiler error.
- Fix the RTL or testbench.
- Compile again.

Step 7:
Run the simulation using run_simulation.

Step 8:
If the simulation fails:
- Analyze the simulation output.
- Identify the likely RTL or testbench problem.
- Fix it.
- Compile again.
- Re-run simulation.

Step 9:
Never claim PASS unless the simulator actually passes.

Keep automatic repair bounded. Do not endlessly modify the design.

The final answer must report:

- Hardware interpretation
- Architecture
- Generated files
- Compilation result
- Simulation result
- Tests performed
- Pass/fail information
- Remaining limitations

IMPORTANT:
A Python algorithm is not necessarily directly synthesizable.
For example, recursion, dynamic lists, file operations, web requests,
and arbitrary software loops may need to be replaced with actual hardware
structures such as registers, memories, FSMs, counters, and combinational
logic.
""",
)


# ============================================================
# 6. API INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description=(
            "Hardware specification or Python algorithm to convert "
            "into Verilog/SystemVerilog."
        )
    )


# ============================================================
# 7. FORMAT USER INPUT
# ============================================================

def format_for_agent(x) -> dict:
    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            (
                "user",
                f"""
Convert the following specification/Python algorithm into
synthesizable SystemVerilog.

USER REQUEST:

{user_input}

Required:
- Generate design.sv
- Generate tb.sv
- Make the testbench self-checking
- Compile the design
- Run the simulation
- If errors occur, debug and repair them
- Do not claim PASS unless simulation actually passes
""",
            )
        ]
    }


# ============================================================
# 8. EXTRACT FINAL AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    # Some agent versions return nested state.
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if not messages:
        return str(agent_output)

    # Find the last useful text message.
    for message in reversed(messages):
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content

        # Some LangChain messages may contain structured content blocks.
        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            if text_parts:
                return "\n".join(text_parts)

    return str(agent_output)


# ============================================================
# 9. LANGCHAIN RUNNABLE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str,
)


# ============================================================
# 10. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="AI Verilog Agent",
    version="1.0.0",
    description=(
        "AI agent that converts hardware specifications/Python algorithms "
        "into SystemVerilog, generates a testbench, compiles and simulates it."
    ),
)

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
)


@app.get("/")
def root():
    return {
        "name": "AI Verilog Agent",
        "status": "running",
        "endpoint": "/agent",
        "description": (
            "Send a hardware specification or Python algorithm "
            "to generate and verify SystemVerilog."
        ),
    }


@app.get("/files")
def files():
    return {
        "files": sorted(p.name for p in WORKDIR.iterdir())
    }


# ============================================================
# 11. START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
