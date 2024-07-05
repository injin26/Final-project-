# Final-project-
Hello! My name is Inji Novruzlu, and for my project with NVIDIA's Jetson Nano, I made a Traffic Sign Recognition System. The main purpose of this is to help the drivers understand when they should stop, be cautious, and go, which can ultimately help reduce the number of car accidents. By testing images of traffic signs, the recognizer will share what the sign means for the driver, controlling their speed and making the road safer for everyone.

Project Structure: please check master branch for code and dataset files
- Final-Project-
- --> DATA
- ----> 0
- ----> 1
- ----> .. #up to 57 folders with images
- --> TEST
- --> labels.csv #CSV file with labels for traffic signs
- --> loadingData.py #loads and preprocesses data
- --> testingModel.py #creates custom traffic sign recognition model
- --> trainingModel.py #trains the model and evaluates it
- --> usingInference.py #understands the test images through jetson-inference library usage

Features:
- CNN model (custom)
- data processing
- model training and testing
- testing images

Sample Images:
![011_0003_j](https://github.com/injin26/Final-project-/assets/160586237/42fef122-e479-4386-b6dc-b51ae4863953)
![035_0017_j](https://github.com/injin26/Final-project-/assets/160586237/536844c4-6c36-451b-8f25-1e00b5a26a84)

Installation/Reproduction:
- clone repository
- install required packages (check for requirements via imports for each python file)
- download the dataset and create a repository for it
- train and then test the model

Link to demonstration video:
