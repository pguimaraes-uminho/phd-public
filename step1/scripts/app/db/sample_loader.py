"""Load the denormalized sample CSVs and project them into per-entity tables.

The CSVs (passenger_flight_data_erd_a.csv / _erd_b.csv) are the single source of
truth for instance data. LLM scenarios receive the raw CSV text (most token-
efficient); the non-LLM baseline needs per-entity tables with ERD attribute
names, produced here via an explicit column map. Entities that appear in two
roles (e.g. an airport as both departure and arrival) are projected from each
role and deduplicated by primary key.
"""
from __future__ import annotations

import csv
import os
from typing import Any

# Repository root (scripts/app/db/ -> up 3). Inputs live in top-level data/.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

CSV_A = os.path.join(_ROOT, "data", "instances_erd_a.csv")
CSV_B = os.path.join(_ROOT, "data", "instances_erd_b.csv")

# entity -> list of projections; each projection maps ERD attribute -> CSV column.
COLMAP_A: dict[str, list[dict[str, str]]] = {
    "Customer": [{"passport": "passport", "name": "name", "surname": "surname"}],
    "Seat": [{"passport": "passport", "flight_number": "flight_number",
              "flight_date": "flight_date", "seat_number": "seat_number"}],
    "Flight": [{"flight_number": "flight_number", "flight_date": "flight_date",
                "airline_code": "airline_code", "airport_arrival": "arrival_airport_code",
                "airport_departure": "departure_airport_code", "belt_number": "baggage_belt",
                "license_number": "airplane_license"}],
    "Airplane": [{"license_number": "airplane_license", "airplane_model": "airplane_model"}],
    "Location": [
        {"location_code": "departure_location_code", "city": "departure_city"},
        {"location_code": "arrival_location_code", "city": "arrival_city"},
    ],
    "Status": [{"status_code": "status_code", "status": "status"}],
    "FlightStatus": [{"status_code": "status_code", "flight_number": "flight_number",
                      "flight_date": "flight_date", "date": "status_date", "hour": "status_hour"}],
    "Airline": [{"airline_code": "airline_code", "name": "airline_name"}],
    "Airport": [
        {"airport_code": "departure_airport_code", "name": "departure_airport_name",
         "location_code": "departure_location_code"},
        {"airport_code": "arrival_airport_code", "name": "arrival_airport_name",
         "location_code": "arrival_location_code"},
    ],
    "BaggageBelt": [{"belt_number": "baggage_belt", "airport_code": "arrival_airport_code"}],
    "CrewMember": [{"crew_license": "crew_license", "role": "crew_role"}],
    "FlightCrew": [{"flight_number": "flight_number", "flight_date": "flight_date",
                    "crew_license": "crew_license"}],
}

COLMAP_B: dict[str, list[dict[str, str]]] = {
    "Passenger": [{"pax_id": "pax_id", "full_name": "full_name",
                   "passport_no": "passport_no", "email": "email"}],
    "Booking": [{"booking_ref": "booking_ref", "pax_id": "pax_id", "flight_id": "flight_id",
                 "seat": "seat", "fare_class": "fare_class", "booking_date": "booking_date"}],
    "Flight": [{"flight_id": "flight_id", "flight_no": "flight_no", "dep_datetime": "dep_datetime",
                "arr_datetime": "arr_datetime", "carrier_code": "carrier_code",
                "aircraft_reg": "aircraft_reg", "origin_iata": "origin_iata",
                "dest_iata": "dest_iata", "status": "flight_status"}],
    "Airline": [{"carrier_code": "carrier_code", "carrier_name": "carrier_name",
                 "country": "carrier_country"}],
    "Aircraft": [{"aircraft_reg": "aircraft_reg", "type_code": "type_code",
                  "seats_capacity": "seats_capacity"}],
    "AircraftType": [{"type_code": "type_code", "model_name": "model_name",
                      "manufacturer": "manufacturer"}],
    "Airport": [
        {"iata_code": "origin_iata", "airport_name": "origin_airport_name",
         "city": "origin_city", "country": "origin_country"},
        {"iata_code": "dest_iata", "airport_name": "dest_airport_name",
         "city": "dest_city", "country": "dest_country"},
    ],
}


def load_csv_text(path: str) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        return f.read().strip()


def csv_to_tables(path: str, colmap: dict[str, list[dict[str, str]]], erd: Any) -> list[dict[str, Any]]:
    """Project the flat CSV into per-entity tables, deduped by each entity's PK."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pk_by_entity = {ent.name: list(getattr(ent, "primary_key", []) or []) for ent in erd.entities}

    tables: list[dict[str, Any]] = []
    for entity, projections in colmap.items():
        pk = pk_by_entity.get(entity) or []
        seen: set = set()
        out_rows: list[dict[str, Any]] = []
        for proj in projections:
            for r in rows:
                rec = {attr: r.get(col, "") for attr, col in proj.items()}
                key = tuple(rec.get(k, "") for k in pk) if pk else tuple(sorted(rec.items()))
                if key in seen:
                    continue
                seen.add(key)
                out_rows.append(rec)
        tables.append({"table": entity, "rows": out_rows})
    return tables
