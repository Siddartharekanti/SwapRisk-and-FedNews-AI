"""Project CLI; `python -m swapscribe.cli --help`."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .paths import ARTIFACTS, DATA, ensure_dirs
from .demo import generate_demo
from .data_sources import fetch_fed_news, fetch_phrasebank
from .market import fetch_treasury, load_rates
from .risk import load_portfolio, risk_report, save_report
from .backtest import walk_forward_backtest
from .sentiment import train_sentiment, load_sentiment_model
from .commentary import load_news, retrieve, build_commentary


def paths(mode: str):
    if mode == "demo":
        return DATA / "demo" / "market_rates.csv", DATA / "demo" / "news.csv"
    files = sorted((DATA / "real").glob("treasury_*.csv"))
    if not files:
        raise FileNotFoundError("No real market data. Run fetch-market --year 2025 (or another year).")
    news = DATA / "real" / "fed_news.csv"
    return files[-1], news


def _prepare(mode):
    if mode == "demo" and not (DATA / "demo" / "market_rates.csv").exists():
        generate_demo(DATA / "demo")
    market_file, news_file = paths(mode)
    rates = load_rates(market_file)
    return rates, news_file


def run_risk(mode, as_of=None, max_hedge=30.):
    rates, _ = _prepare(mode)
    portfolio_path = (DATA / "real" / "portfolio.csv") if (DATA / "real" / "portfolio.csv").exists() else (DATA / "demo" / "portfolio.csv")
    if not portfolio_path.exists():
        generate_demo(DATA / "demo")
    report = risk_report(rates, load_portfolio(portfolio_path), as_of=as_of, max_abs_millions=max_hedge)
    report["source_mode"] = mode
    report["portfolio_source"] = "institutional benchmark portfolio; standard G10 tenors"
    return report


def run_commentary(mode, query, as_of=None, model_file=None):
    rates, news_file = _prepare(mode)
    if not news_file.exists():
        raise FileNotFoundError(f"News missing at {news_file}. Run fetch-news or choose demo mode.")
    cutoff = as_of or rates.date.max()
    news = load_news(news_file)
    default_model = ARTIFACTS / "sentiment_real.joblib" if (ARTIFACTS / "sentiment_real.joblib").exists() else ARTIFACTS / "sentiment_demo.joblib"
    model_file = model_file or default_model
    model = load_sentiment_model(model_file) if model_file.exists() else None
    ranked = retrieve(news, query, cutoff, k=6)
    report = build_commentary(rates, ranked, cutoff, model)
    report["mode"] = mode
    report["sentiment_model"] = model_file.name if model is not None else "not trained"
    return report


def main(argv=None):
    ensure_dirs()
    parser = argparse.ArgumentParser(prog="swapscribe", description="CPU-first Global Markets research workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Generate entirely synthetic, repeatable offline fixtures")
    p = sub.add_parser("fetch-market", help="Download official U.S. Treasury par yields (one calendar year)")
    p.add_argument("--year", type=int, default=2025)
    sub.add_parser("fetch-news", help="Download current Federal Reserve press-release RSS metadata")
    p = sub.add_parser("fetch-sentiment", help="Download Financial PhraseBank (noncommercial license)")
    p.add_argument("--agreement", choices=["50", "66", "75", "all"], default="75")
    p = sub.add_parser("train", help="Train/evaluate TF-IDF+logistic sentiment from demo or Financial PhraseBank")
    p.add_argument("--source", choices=["demo", "real"], default="demo")
    p = sub.add_parser("risk", help="Value synthetic swaps and optimize hedge")
    p.add_argument("--mode", choices=["demo", "real"], default="demo")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; strictly no future observations")
    p.add_argument("--max-hedge", type=float, default=30., help="Max absolute millions per hedge instrument")
    p = sub.add_parser("commentary", help="Generate grounded market note + headline evidence")
    p.add_argument("--mode", choices=["demo", "real"], default="demo")
    p.add_argument("--query", default="interest rates treasury yields monetary policy")
    p.add_argument("--as-of", default=None)
    p.add_argument("--model", choices=["demo", "real", "none"], default="demo")
    p = sub.add_parser("backtest", help="Time-forward historical first-order hedge assessment")
    p.add_argument("--mode", choices=["demo", "real"], default="demo")
    p.add_argument("--lookback", type=int, default=70)
    p.add_argument("--holding-days", type=int, default=5)
    p = sub.add_parser("evaluate", help="Run risk and available sentiment evaluations")
    p.add_argument("--mode", choices=["demo", "real"], default="demo")
    p = sub.add_parser("all", help="One-command workflow: train, risk report, commentary, backtest")
    p.add_argument("--mode", choices=["real", "demo"], default="real")
    args = parser.parse_args(argv)

    if args.command == "demo":
        print(json.dumps({k: str(v) for k, v in generate_demo(DATA / "demo").items()}, indent=2))
    elif args.command == "fetch-market":
        print("Saved official Treasury data:", fetch_treasury(args.year, DATA / "real" / f"treasury_{args.year}.csv"))
    elif args.command == "fetch-news":
        print("Saved Fed press-release RSS:", fetch_fed_news(DATA / "real" / "fed_news.csv"))
    elif args.command == "fetch-sentiment":
        print("Saved noncommercial Financial PhraseBank:", fetch_phrasebank(DATA / "real" / "sentiment_train.csv", agreement=args.agreement))
    elif args.command == "train":
        if args.source == "demo":
            _prepare("demo")
        csv = DATA / args.source / "sentiment_train.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Missing {csv}. Run fetch-sentiment for real or demo for demo.")
        result = train_sentiment(csv, ARTIFACTS / f"sentiment_{args.source}.joblib",
                                 ARTIFACTS / f"sentiment_{args.source}_eval.json", dataset_type=args.source)
        print(json.dumps(result, indent=2))
    elif args.command == "risk":
        report = run_risk(args.mode, args.as_of, args.max_hedge)
        out = save_report(report, ARTIFACTS / f"risk_{args.mode}.json")
        print(f"Saved {out}\n" + json.dumps(report, indent=2))
    elif args.command == "commentary":
        m = ARTIFACTS / f"sentiment_{args.model}.joblib" if args.model != "none" else Path("non-existent-model")
        report = run_commentary(args.mode, args.query, args.as_of, m)
        out = save_report(report, ARTIFACTS / f"commentary_{args.mode}.json")
        print(f"Saved {out}\n" + json.dumps(report, indent=2))
    elif args.command == "backtest":
        rate_history, _ = _prepare(args.mode)
        portfolio_path = (DATA / "real" / "portfolio.csv") if (DATA / "real" / "portfolio.csv").exists() else (DATA / "demo" / "portfolio.csv")
        result = walk_forward_backtest(rate_history, load_portfolio(portfolio_path),
                                       lookback=args.lookback, holding_days=args.holding_days)
        out = save_report(result, ARTIFACTS / f"backtest_{args.mode}.json")
        print(f"Saved {out}\n" + json.dumps({k:v for k,v in result.items() if k != "oos_pnl_series"}, indent=2))
    elif args.command == "evaluate":
        report = run_risk(args.mode)
        print(json.dumps({"risk_model": report, "sentiment_evaluation": _eval_if_exists(args.mode)}, indent=2))
    elif args.command == "all":
        if args.mode == "real":
            market_file, news_file = paths("real")
            sentiment_csv = DATA / "real" / "sentiment_train.csv"
            if not sentiment_csv.exists():
                raise FileNotFoundError("Missing real sentiment data at data/real/sentiment_train.csv. Run fetch-sentiment first.")
            sentiment_metrics = train_sentiment(sentiment_csv, ARTIFACTS / "sentiment_real.joblib",
                                                ARTIFACTS / "sentiment_real_eval.json", dataset_type="real")
            risk = run_risk("real")
            commentary = run_commentary("real", "interest rates treasury yields monetary policy",
                                        model_file=ARTIFACTS / "sentiment_real.joblib")
            rates = load_rates(market_file)
            portfolio_path = DATA / "demo" / "portfolio.csv"
            if not portfolio_path.exists():
                generate_demo(DATA / "demo")
            backtest = walk_forward_backtest(rates, load_portfolio(portfolio_path))
            save_report(backtest, ARTIFACTS / "backtest_real.json")
            save_report(risk, ARTIFACTS / "risk_real.json")
            save_report(commentary, ARTIFACTS / "commentary_real.json")
            print(json.dumps({
                "status": "success",
                "mode": "real (official U.S. Treasury + Fed RSS + Financial PhraseBank)",
                "as_of": risk.get("as_of"),
                "portfolio_pv_usd": risk["portfolio_pv_usd"],
                "parallel_dv01_usd_per_bp": risk["parallel_dv01_usd_per_bp"],
                "risk_rms_before": risk["rms_scenario_pnl_usd_before"],
                "risk_rms_after": risk["rms_scenario_pnl_usd_after"],
                "worst_scenario_before": risk["worst_scenario_pnl_usd_before"],
                "worst_scenario_after": risk["worst_scenario_pnl_usd_after"],
                "oos_rms_unhedged": backtest["oos_rms_pnl_usd_unhedged"],
                "oos_rms_optimized": backtest["oos_rms_pnl_usd_optimized"],
                "oos_rms_parallel_only": backtest["oos_rms_pnl_usd_parallel_only"],
                "sentiment_accuracy": sentiment_metrics["test_accuracy"],
                "sentiment_macro_f1": sentiment_metrics["test_macro_f1"],
                "model_metrics": sentiment_metrics,
                "artifacts": str(ARTIFACTS)
            }, indent=2))
        else:
            generate_demo(DATA / "demo")
            train_sentiment(DATA / "demo" / "sentiment_train.csv", ARTIFACTS / "sentiment_demo.joblib",
                            ARTIFACTS / "sentiment_demo_eval.json", dataset_type="demo")
            risk = run_risk("demo")
            commentary = run_commentary("demo", "interest rates treasury yields monetary policy")
            backtest = walk_forward_backtest(load_rates(DATA / "demo" / "market_rates.csv"),
                                             load_portfolio(DATA / "demo" / "portfolio.csv"))
            save_report(backtest, ARTIFACTS / "backtest_demo.json")
            save_report(risk, ARTIFACTS / "risk_demo.json")
            save_report(commentary, ARTIFACTS / "commentary_demo.json")
            print(json.dumps({"status": "success", "demo_is_synthetic": True,
                              "risk_rms_before": risk["rms_scenario_pnl_usd_before"],
                              "risk_rms_after": risk["rms_scenario_pnl_usd_after"],
                              "oos_rms_unhedged": backtest["oos_rms_pnl_usd_unhedged"],
                              "oos_rms_optimized": backtest["oos_rms_pnl_usd_optimized"],
                              "model_metrics": _eval_if_exists("demo"),
                              "artifacts": str(ARTIFACTS)}, indent=2))


def _eval_if_exists(mode):
    path = ARTIFACTS / f"sentiment_{mode}_eval.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not trained"}


if __name__ == "__main__":
    main()
