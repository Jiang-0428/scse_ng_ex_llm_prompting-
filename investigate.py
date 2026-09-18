## Import the necessary modules
import json
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result

## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.

def build_prompt(description, available_items):
    system_prompt = """
STRICT RULE: ONLY RETURN RAW JSON. NO comments, no explanations, no extra words.
The JSON must have two keys: "matches" (list of item ID strings), "confidence".
confidence value must be one of: "HIGH", "MEDIUM", "LOW".
Use only the items from the provided lost item dataset.
Include ALL matching unclaimed items.
If no matches: {"matches": [], "confidence": "LOW"}

Example correct output:
{"matches":["F102"],
"confidence":"MEDIUM"}
"""

    items_json = json.dumps(available_items)
    user_prompt = f"""User lost‑item description: {description}

Available items:
{items_json}

Find all possible matches and output only the required JSON."""
    return system_prompt, user_prompt

    
## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ]
    )
    return response["message"]["content"]

## Logic to parse the response from Qwen and return the result. 
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON object found in model response")
    return result


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False

    valid_confidence = {"LOW", "MEDIUM", "HIGH"}
    if not isinstance(result["confidence"], str) or result["confidence"] not in valid_confidence:
        return False

    valid_ids = {x["id"] for x in available_items}
    cleaned = [item_id for item_id in result["matches"] if item_id in valid_ids]
    result["matches"] = cleaned

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
""" 
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """


## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-"*50)
    print(f"Confidence: {result['confidence']}")
    if not result["matches"]:
        print("\nPossible matches: []")
    else:
        print("\nPossible matches:")
        lookup = {item["id"]:item for item in available_items}
        for iid in result["matches"]:
            it = lookup[iid]
            print(f"\nID: {it['id']}")
            print(f"Item: {it['item']}")
            print(f"Color: {it['color']}")
            print(f"Location: {it['location']}")
            print(f"Date found: {it['date']}")
    print("\nResult saved to output/match_result.json")
    

## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("="*50)
    try:
        all_items = load_items("found_items.json")
    except FileNotFoundError:
        print("Error: found_items.json file not found")
        return
    unclaimed = get_unclaimed_items(all_items)
    description = input("\nDescribe the item you lost: ")
    print("\nSearching for possible matches...")
    sys_prompt, usr_prompt = build_prompt(description, unclaimed)
    try:
        raw = ask_qwen(sys_prompt, usr_prompt)
    except Exception as e:
        print(f"Error communicating with ollama/qwen: {e}")
        return
    try:
        res = parse_response(raw)
    except Exception as e:
        print(f"Fallback due to parsing error: {e}")
        res = {"matches": [], "confidence": "LOW"}
        
    if not validate_result(res, unclaimed):
        print("Error: Invalid result format or invalid item IDs.")
        return
    display_matches(res, unclaimed)
    save_result(res, "output/match_result.json")

if __name__ == "__main__":
    main()