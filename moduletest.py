import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

newmodule = models.resnet50()
model = torch.load('C:/Users/user/skincancer/save/20230915.pth', map_location=torch.device('cpu'))
# 加载已经训练好的模型权重到newmodule
newmodule.load_state_dict(model)
newmodule.eval()  # 将模型设置为评估模式


# 定义图像预处理步骤
preprocess = transforms.Compose([
   transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(contrast=0.2),# 定義對比度增強轉換
    transforms.RandomRotation(degrees=10),# 定義隨機旋轉轉換
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 調整色度、亮度、飽和度、對比度
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# 加载测试图像
image = Image.open('./20.jpg')  # 替换为您的测试图像文件路径
input_image = preprocess(image)
input_batch = input_image.unsqueeze(0)  # 添加批次维度

# 使用模型进行推断
with torch.no_grad():
    output = newmodule(input_batch)

# 在输出中查找预测结果
# 这取决于您的模型和任务，通常涉及将输出映射到类别或结果的后处理步骤
# 例如，如果您有一个分类模型，您可以使用softmax来获取概率分布
# 并找到最高概率的类别
output_probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(output_probabilities).item()

# 打印预测结果
print(f"Predicted class: {predicted_class}")

# 可以进一步将预测类别与类别标签进行映射
# 这需要您有一个类别到标签的映射
# 例如，如果您有一个名为 class_labels 的列表，其中包含类别名称
# 那么您可以使用 predicted_class 来查找对应的类别名称
# predicted_class_name = class_labels[predicted_class]
# print(f"Predicted class: {predicted_class_name}")
