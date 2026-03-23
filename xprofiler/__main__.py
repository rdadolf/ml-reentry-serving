import argparse
import sys


def cmd_summary(args):
    from xprofiler import summary, trace

    try:
        config = summary.load_model_config(args.model)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    t = trace.load(args.trace)
    result = summary.summarize(t, config)
    print(summary.to_json(result))


def main():
    parser = argparse.ArgumentParser(prog="xprofiler")
    sub = parser.add_subparsers(dest="command")

    p_summary = sub.add_parser("summary", help="Produce per-block architectural summary")
    p_summary.add_argument("trace", help="Path to Chrome trace JSON (.json or .json.gz)")
    p_summary.add_argument(
        "--model",
        default="llama",
        help="Model config name (default: llama). See xprofiler/models/",
    )

    args = parser.parse_args()
    if args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
