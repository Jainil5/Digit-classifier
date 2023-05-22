import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
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

    #train

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


    #test on img of 28x28

    with open("digit-classifier/model_state.pt","rb") as f:
        clf.load_state_dict(load(f))

    
    image = cv2.imread("digit-classifier/digits/3.png")
    res = cv2.resize(image,(128,128))
    res2 = cv2.resize(res,(64,64))
    res3 = cv2.resize(res2,(32,32))
    gray = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)

    img_tensor = ToTensor()(res3).reshape(32,1,3,3)

    print(torch.argmax(clf(img_tensor)).item())