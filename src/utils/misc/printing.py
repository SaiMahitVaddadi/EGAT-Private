def print_pretty_dict(d):
    # Determine the longest key to format nicely
    max_key_length = max(len(str(key)) for key in d.keys())
    # Print header
    print("Key".ljust(max_key_length + 4) + "Value")
    print("-" * (max_key_length + 10))
    # Print each key-value pair in a nicely formatted way
    for key, value in d.items():
        print(f"{str(key).ljust(max_key_length + 4)}{value}")