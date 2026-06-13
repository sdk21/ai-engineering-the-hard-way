"""
Demo: Sequential Pipeline
Usage:
    python demo.py --mock
    python demo.py --real [--stages 2|3|4]
"""

import argparse
import os
import sys

from experiment import (
    EXAMPLE_INPUTS, PIPELINE_STAGES, PipelineResult, PipelineRun,
    stage_prompt, mock_pipeline_run,
)


def mock_demo() -> None:
    print("\n=== Sequential Pipeline Demo [MOCK] ===")
    run = mock_pipeline_run()
    run.display()


def real_demo(num_stages: int) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    stages = PIPELINE_STAGES[:num_stages]
    print(f"\n=== Sequential Pipeline Demo [REAL, stages={num_stages}] ===")
    print(f"Pipeline: {' → '.join(s.name for s in stages)}")
    print("\nExample inputs:")
    for i, inp in enumerate(EXAMPLE_INPUTS, 1):
        print(f"  {i}. {inp[:70]}")
    print("\nEnter feedback text (or number 1-4) or 'quit':\n")

    while True:
        try:
            inp = input("Input: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not inp or inp.lower() == "quit":
            break
        text = EXAMPLE_INPUTS[int(inp)-1] if inp.isdigit() and 1 <= int(inp) <= 4 else inp

        run = PipelineRun(input_text=text)
        current_text = text

        for stage in stages:
            print(f"\n  [{stage.name}] {stage.description}...")
            r = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=stage.system,
                messages=[{"role": "user", "content": stage_prompt(stage, current_text, run.stages)}],
            )
            output = r.content[0].text.strip()
            result = PipelineResult(stage_name=stage.name, input_text=current_text, output_text=output)
            run.stages.append(result)
            current_text = output
            print(f"    → {output[:100]}")

        print(f"\n  Final output:\n  {run.final_output()}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--real", action="store_true")
    parser.add_argument("--stages", type=int, default=4, choices=[2, 3, 4])
    args = parser.parse_args()
    if args.real and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    if args.mock:
        mock_demo()
    else:
        real_demo(num_stages=args.stages)
