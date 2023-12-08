import argparse
import dataset as D
import json
import time

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="")
    # Add the arguments
    parser.add_argument("--function", type=str, required=False, default='srl', help="function to run. Options: prepare_data train, test, predict")
    parser.add_argument("--config", type=str, required=False, help="path to the config file. If none is provided, the default config for given function will be used")
    # Parse the arguments
    args = parser.parse_args()

    # If a config path is given, load it
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)

    if args.function == "prepare_data":
        # load default config if none is given
        with open("../config/prepare_data.json", "r") as f:
            config = json.load(f)
        
        # function
        D.prepare_data(config)
    
    elif args.function == "srl":
        # load default config if none is given
        with open("../config/srl.json", "r") as f:
            config = json.load(f)
        
        # function
        D.srl(config)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"DONE main() in {time.time()-start_time:.2f}s")