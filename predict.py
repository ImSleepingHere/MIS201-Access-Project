"""Friendly used-car price estimator.  Run:  python predict.py
(Run train.py once first to create model.joblib.)"""
import sys
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")


def pick(prompt, options, default=None):
    """Ask until the answer matches one of the options (case-insensitive) or a list number."""
    lookup = {o.lower(): o for o in options}
    while True:
        hint = f" [Enter = {default}]" if default else ""
        a = input(f"{prompt}{hint}: ").strip()
        if not a and default:
            return default
        if a.lower() in lookup:
            return lookup[a.lower()]
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1]
        print("  Please choose one of:", ", ".join(options))


def number(prompt, lo, hi, cast=int, default=None):
    while True:
        hint = f" [Enter = {default}]" if default is not None else ""
        a = input(f"{prompt}{hint}: ").strip().replace(",", "")
        if not a and default is not None:
            return default
        try:
            v = cast(a)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Please enter a number between {lo:,} and {hi:,}.")


def main():
    try:
        bundle = joblib.load("model.joblib")
    except FileNotFoundError:
        sys.exit("model.joblib not found - run  python train.py  first.")
    model, meta = bundle["model"], bundle["meta"]
    d = meta["defaults"]

    print("\n=== Saudi Used-Car Price Estimator ===")
    print("Answer a few questions and get an estimated price in SAR.\n")
    while True:
        makes = sorted(meta["makes"])
        print("Makes:", ", ".join(makes))
        make = pick("Make", makes)
        models = meta["makes"][make]
        print(f"{make} models:", ", ".join(models))
        car_model = pick("Model", models)
        year = number("Year", meta["year_range"][0], meta["year_range"][1] + 1)
        mileage = number("Mileage (km)", 0, 1_000_000)
        print("Regions:", ", ".join(meta["regions"]))
        region = pick("Region", meta["regions"], default=meta["regions"][0])
        engine = number("Engine size (litres, e.g. 2.5)", 0.5, 10.0, float, default=2.0)
        gear = pick("Gearbox (Automatic/Manual)", meta["gears"], default=d["GearType"])
        fuel = pick("Fuel (Gas/Diesel/Hybrid)", meta["fuels"], default=d["FuelType"])
        options = pick("Options level (Standard/Semi Full/Full)", meta["options"], default=d["Options"])

        car = pd.DataFrame([{
            "MakeName": make, "Country": meta["make_country"][make], "Model": car_model,
            "FuelType": fuel, "GearType": gear, "Options": options, "RegionName": region,
            "Year": year, "Mileage": mileage, "EngineSize": engine,
        }])
        price = float(model.predict(car)[0])
        print(f"\n  >>> Estimated price for a {year} {make} {car_model}: {round(price, -2):,.0f} SAR")
        print("      (typical error is about +/- 9,300 SAR)\n")
        if input("Estimate another car? (y/n): ").strip().lower() not in ("y", "yes"):
            print("Thanks - goodbye!")
            break
        print()


if __name__ == "__main__":
    main()
