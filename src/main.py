import sys
import os

def main():
    print("Road Sign Recognition Application")
    print("---------------------------------")
    print("This application uses a Multi-Scale CNN based on Sermanet and LeCun architecture.")
    print("Dataset: German Traffic Sign Recognition Benchmark (GTSRB)")
    print("")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/main.py train              - Train the model")
        print("  python src/main.py predict <path>      - Predict sign in image")
        print("  python src/main.py gui                - Launch the graphical user interface")
        return

    command = sys.argv[1].lower()
    
    if command == "train":
        from train import train_model
        train_model()
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Please provide an image path.")
            return
        from predict import predict
        img_path = sys.argv[2]
        result = predict(img_path)
        if result:
            print(f"Predicted Traffic Sign: {result}")
    elif command == "gui":
        from gui import main as gui_main
        gui_main()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()