"""Command line: data, train, eval, route, serve."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import DATA_DIR, get_settings


def cmd_data(args: argparse.Namespace) -> None:
    from .data import prepare

    print(json.dumps(prepare(args.train, args.val, args.test), indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from .data import read_jsonl
    from pathlib import Path

    from .model import load_model, load_tokenizer, pick_device, set_threads
    from .train import TrainConfig, train

    settings = get_settings()
    base_model = args.base_model or settings.base_model
    out_dir = Path(args.out) if args.out else settings.adapter_dir
    set_threads(settings.threads)
    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, rank=args.rank, alpha=2 * args.rank,
        learning_rate=args.lr, eval_every=args.eval_every, micro_batch_size=args.micro_batch_size,
    )
    rows = read_jsonl(DATA_DIR / "train.jsonl")[: args.limit or None]
    tokenizer = load_tokenizer(base_model)
    model = load_model(base_model, device=args.device or pick_device())
    summary = train(model, tokenizer, rows, read_jsonl(DATA_DIR / "val.jsonl"), out_dir, cfg,
                    log=lambda line: print(line, flush=True))
    print(f"saved the LoRA add-on to {out_dir} after {summary['training_minutes']} minutes")


def cmd_route(args: argparse.Namespace) -> None:
    from .providers import build_provider
    from .service import route

    provider = build_provider(get_settings())
    for message in args.message:
        routed = asdict(route(provider, message))
        print(json.dumps(routed, indent=2, ensure_ascii=False))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("triage.api:app", host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download the public dataset and write the splits")
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--test", type=int, default=1000)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="fine-tune the LoRA add-on (CPU, CUDA or Apple GPU)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=0, help="split each batch to save memory")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=0, help="use only the first N training examples")
    p.add_argument("--base-model", default="", help="Hub id of the base model (default: TRIAGE_BASE_MODEL)")
    p.add_argument("--out", default="", help="folder for the adapter (default: TRIAGE_ADAPTER_DIR)")
    p.add_argument("--device", default="", help="cpu, cuda or mps (default: the fastest available)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("route", help="route one or more messages")
    p.add_argument("message", nargs="+")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("serve", help="start the web API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8540)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
