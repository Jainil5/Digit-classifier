import torch
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import cv2

train = datasets.MNIST(root="data",download=True,train=True,transform=ToTensor())
dataset = DataLoader(train,32)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model =nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )
    def forward(self,x):
        return self.model(x)


#Instance of neural net

clf = ImageClassifier().to("cpu")
optimizer = Adam(clf.parameters(),lr= 1e-3)
lossfn = nn.CrossEntropyLoss()

# Training flow

if __name__ == "__main__":

    #### train

    # for epoch in range(5):
    #     for batch in dataset:
    #         X,y = batch
    #         X,y = X.to("cpu"),y.to("cpu")
    #         yhat = clf(X)
    #         loss = lossfn(yhat,y)
    
    #         #Apply backprop
    
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Epoch {epoch} X Loss: {loss.item()}")
    
    # with open("model_state.pt","wb") as f:
    #     save(clf.state_dict(),f)


    ####    test on img of 28x28

    with open("model_state.pt","rb") as f:
        clf.load_state_dict(load(f))

    image_path = "img9.jpg"
    img = Image.open(image_path)
    image = cv2.imread(image_path)

    resize = cv2.resize(image,(480,480))
    cv2.imshow("Image",resize)
    img_tensor = ToTensor()(img).unsqueeze(0).to("cpu")
    
    print("The digit is: ",torch.argmax(clf(img_tensor)).item())

    cv2.waitKey(0)