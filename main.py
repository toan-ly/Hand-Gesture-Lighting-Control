from src.deployment.detect_simulation import LightGesture

def main():
    model_path = './models/model_27-11 12_38_NeuralNetwork_best'
    light = LightGesture(model_path, device=False)
    light.run()
    
if __name__ == "__main__":
    main()