####################BOT SETUP##########################
from pytorch_bot import bot

telegram_token = "YOUR TOKEN HERE"   # replace TOKEN with your bot's token it must be string

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = None   # replace None with your telegram user id (integer):
bot = bot(token=telegram_token, user_id=telegram_user_id)
bot.activate_bot()

#########################PYTORCH########################
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:- ",device,end='\n')
# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
	train=True, transform=transforms.ToTensor(),  download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', 
	train=False, transform=transforms.ToTensor(),  download=True)

# Data Loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
		batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
		batch_size=batch_size, shuffle=False)

# Neural Network Class
class NeuralNet(nn.Module):
	"""FeedForward Neural Network with one hidden layer
	"""
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out 


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def test(message):
	global status_message

	with torch.no_grad():
		correct = total = 0
		for images, labels in test_loader:
			images = images.reshape(-1,28*28).to(device)
			labels = labels.to(device)

			# Forward pass
			outputs = model(images)
			loss = loss_func(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	message += '\nTest: Loss: {:.4f} Accuracy: {:.2f}'.format(loss, (100*correct)/total)
	return message


# Training 
print("Training Started....!!")
print()
for epoch in range(num_epochs):
	cumulative_loss =  0.0
	total = correct = 0
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1,28*28).to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = loss_func(outputs, labels)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		cumulative_loss += loss.item()
		# Backprop and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	
	
	message = 'Train: Epoch [{}/{}], Loss: {:.4f} Accuracy: {:.2f}'.format(epoch+1, num_epochs, cumulative_loss/total, (100*correct)/total)
	message = test(message)
	print(message)
	bot.send_message(message)
	bot.set_status(message)

print("Training finished...!!")
bot.send_message('Training finished..!!')
bot.stop_bot()