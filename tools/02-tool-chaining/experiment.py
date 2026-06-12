"""
Tool Chaining
-------------
Demonstrates multi-step tool use where the output of one tool becomes
the input (or informs the arguments) of the next.

Key concepts:
- Sequential dependency: tool B needs the result of tool A
- The model manages the chain — it decides the order and passes results forward
- Each round-trip sends the full conversation so the model has all context
- The chain terminates when the model can answer from accumulated results

This experiment uses a travel-planning scenario:
  1. look_up_flight(origin, destination) → flight_id, price, duration
  2. get_flight_details(flight_id) → departure_time, airline, seat_options
  3. check_seat_availability(flight_id, seat_class) → available seats
  4. calculate_total_cost(base_price, extras) → total with fees

Each step depends on data returned by the previous one.
"""

import random

# ---------------------------------------------------------------------------
# Fake data stores
# ---------------------------------------------------------------------------

FLIGHTS = {
    "F001": {"origin": "NYC", "destination": "LAX", "price": 320, "duration": "5h 30m"},
    "F002": {"origin": "NYC", "destination": "LAX", "price": 280, "duration": "6h 15m"},
    "F003": {"origin": "SFO", "destination": "ORD", "price": 210, "duration": "4h 45m"},
    "F004": {"origin": "LAX", "destination": "MIA", "price": 190, "duration": "5h 00m"},
}

FLIGHT_DETAILS = {
    "F001": {"airline": "SkyLine Air", "departure": "08:30 AM", "seat_options": ["economy", "business"]},
    "F002": {"airline": "Coast Airways", "departure": "02:15 PM", "seat_options": ["economy"]},
    "F003": {"airline": "MidWest Express", "departure": "11:00 AM", "seat_options": ["economy", "first"]},
    "F004": {"airline": "Sun Shuttle", "departure": "06:45 AM", "seat_options": ["economy", "business"]},
}

SEAT_AVAILABILITY = {
    ("F001", "economy"): 14,
    ("F001", "business"): 3,
    ("F002", "economy"): 0,
    ("F003", "economy"): 22,
    ("F003", "first"): 2,
    ("F004", "economy"): 8,
    ("F004", "business"): 5,
}

SEAT_FEES = {"economy": 0, "business": 150, "first": 300}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def look_up_flight(origin: str, destination: str) -> dict:
    """Find available flights between two cities."""
    matches = [
        {"flight_id": fid, **data}
        for fid, data in FLIGHTS.items()
        if data["origin"].upper() == origin.upper()
        and data["destination"].upper() == destination.upper()
    ]
    if not matches:
        return {"error": f"No flights found from {origin} to {destination}"}
    return {"flights": matches}


def get_flight_details(flight_id: str) -> dict:
    """Get detailed information for a specific flight ID."""
    flight_id = flight_id.upper()
    base = FLIGHTS.get(flight_id)
    details = FLIGHT_DETAILS.get(flight_id)
    if not base or not details:
        return {"error": f"Unknown flight ID: {flight_id}"}
    return {
        "flight_id": flight_id,
        "airline": details["airline"],
        "departure": details["departure"],
        "origin": base["origin"],
        "destination": base["destination"],
        "duration": base["duration"],
        "base_price": base["price"],
        "seat_options": details["seat_options"],
    }


def check_seat_availability(flight_id: str, seat_class: str) -> dict:
    """Check how many seats of a given class are available on a flight."""
    flight_id = flight_id.upper()
    seat_class = seat_class.lower()
    key = (flight_id, seat_class)
    if key not in SEAT_AVAILABILITY:
        return {"error": f"No '{seat_class}' class on flight {flight_id}"}
    count = SEAT_AVAILABILITY[key]
    return {
        "flight_id": flight_id,
        "seat_class": seat_class,
        "seats_available": count,
        "status": "available" if count > 0 else "sold out",
    }


def calculate_total_cost(base_price: int, seat_class: str, num_passengers: int = 1) -> dict:
    """Calculate the total booking cost including class upgrade fees."""
    seat_class = seat_class.lower()
    fee = SEAT_FEES.get(seat_class, 0)
    per_person = base_price + fee
    total = per_person * num_passengers
    return {
        "base_price_per_person": base_price,
        "seat_upgrade_fee": fee,
        "price_per_person": per_person,
        "num_passengers": num_passengers,
        "total_cost": total,
        "currency": "USD",
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "look_up_flight",
        "description": (
            "Search for available flights between two cities. "
            "Returns a list of flights with flight IDs, prices, and durations. "
            "Use this first to find candidate flights before getting details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "3-letter airport code or city name, e.g. 'NYC' or 'SFO'"},
                "destination": {"type": "string", "description": "3-letter airport code or city name, e.g. 'LAX' or 'ORD'"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "get_flight_details",
        "description": (
            "Get full details for a specific flight using its flight ID (e.g. 'F001'). "
            "Returns airline, departure time, duration, base price, and available seat classes. "
            "You must call look_up_flight first to get a valid flight ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "Flight ID returned by look_up_flight, e.g. 'F001'"},
            },
            "required": ["flight_id"],
        },
    },
    {
        "name": "check_seat_availability",
        "description": (
            "Check how many seats of a given class (economy, business, first) are available on a flight. "
            "Returns the count and whether seats are available or sold out. "
            "Use get_flight_details first to confirm which seat classes the flight offers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "Flight ID, e.g. 'F001'"},
                "seat_class": {"type": "string", "description": "Seat class: 'economy', 'business', or 'first'"},
            },
            "required": ["flight_id", "seat_class"],
        },
    },
    {
        "name": "calculate_total_cost",
        "description": (
            "Calculate the total booking cost given a base price, seat class, and number of passengers. "
            "Returns the per-person price, upgrade fee, and total. "
            "Use this after confirming availability to give the user a final price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_price": {"type": "integer", "description": "Base ticket price in USD from get_flight_details"},
                "seat_class": {"type": "string", "description": "Seat class: 'economy', 'business', or 'first'"},
                "num_passengers": {"type": "integer", "description": "Number of passengers (default: 1)"},
            },
            "required": ["base_price", "seat_class"],
        },
    },
]

TOOL_FN = {
    "look_up_flight": look_up_flight,
    "get_flight_details": get_flight_details,
    "check_seat_availability": check_seat_availability,
    "calculate_total_cost": calculate_total_cost,
}


def dispatch_tool(name: str, inputs: dict) -> str:
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    result = fn(**inputs)
    # Return as a readable string for the model
    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        lines = [f"{k}: {v}" for k, v in result.items()]
        return "\n".join(lines)
    return str(result)
