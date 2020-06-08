import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import mnistClass

mnist = mnistClass.mnistClass()

class Nnit(nn.Module):
    def __init__(self,inc,ouc):
        super(Unit,self).__init__()
        self.unit_net = nn.Sequential(nn.Conv2d(inc,ouc,kernel_size=5,padding=1),
                                      nn.BatchNorm2d(ouc),
                                      nn.Relu())
    def forward(self,x):
        return self.unit_net(x)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net = nn.Sequential(Unit(1,32),    #一层卷积32个5*5，输入15*16
                                 Unit(32,32),
                                 Unit(32,32),

                                 nn.MaxPool2d(2),  #8*8

                                 Unit(32,64),
                                 Unit(64,64),
                                 Unit(64,64),

                                 nn.MaxPool2d(2),   #4*4
                                 )
        self.fc1 = nn.Linear(4*4*64,512)
        self.fc2 = nn.Linear(512,300)

    def forward(self,x):
        y=self.net(x)
        y=y.view(-1,512)
        return self.fc2(y)


device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
batch_size = 8
LR = 0.003

net = LeNet().to(device)
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数使用 Adam 自适应优化算法
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
    weight_decay=0.0001,
)


def test():  # 测试集1万张
    test_acc = 0
    module.eval()
    for j, (imgs, labels) in enumerate(test_dataloader):  # 每次处理512张
        if CUDA:
            imgs = imgs.cuda()
            labels = labels.cuda()
        outs = module(imgs)
        # 训练求loss是为了做权重更新，测试里不需要
        _, prediction = torch.max(outs, 1)
        test_acc += torch.sum(prediction == labels)
    test_acc = test_acc.cpu().item() / 10000
    return test_acc


def train(num_epoch):  # 训练集6万张
    if os.path.exists(param_path):
        module.load_state_dict(torch.load(param_path))
    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0
        module.train()
        for i, (imgs, labels) in enumerate(train_dataloader):  # 每次处理512张
            # print('labels:',labels)#每个标签对应一个0-9的数字
            if CUDA:
                imgs = imgs.cuda()
                labels = labels.cuda()
            outs = module(imgs)
            # print(outs.shape)
            # print('outs:',outs)
            loss = loss_f(outs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('1111',loss)
            # print('2222',loss.data)#tensor且GPU
            # print('3333',loss.cpu())
            # print('4444',loss.cpu().data)#tensor且CPU
            # # print('5555',loss.cpu().data[0])#报错 IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
            # # print('6666',loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
            # print('7777',loss.cpu().detach().numpy())
            # print('8888',loss.cpu().data.numpy())
            # print('9999',loss.cpu().item())
            # print('aaaa',loss.item())#后四者一样，都是把数值取出来
            train_loss += loss.cpu().item() * imgs.size(0)  # imgs.size(0)批次
            '分类问题，常用torch.max(outs,1)得到索引来表示类别'
            _, prediction = torch.max(outs, 1)  # prediction对应每行最大值所在位置的索引值，即0-9
            train_acc += torch.sum(prediction == labels)
            # print(train_acc.cpu().item())

        adjust_lr_rate(epoch)
        train_loss = train_loss / 60000
        train_acc = train_acc.cpu().item() / 60000  # 此处求概率必须用item()把数值取出，否则求出的不是小数

        '每训练完一个epoch，用测试集做一遍评估'
        test_acc = test()
        best_acc = 0
        if test_acc > best_acc:
            best_acc = test_acc
            if os.path.exists(tmp_param_path):
                shutil.copyfile(tmp_param_path, param_path)  # 防权重损坏
            torch.save(module.state_dict(), tmp_param_path)
        print('Epoch:', epoch, 'Train_Loss:', train_loss, 'Train_Acc:', train_acc, 'Test_Acc:', test_acc)


train(1000)

sess=tf.InteractiveSession()

batch_size = 8
kokoko= mnist.getNum()
n_batch = mnist.getNum() // batch_size

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_viaviable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

x=tf.placeholder(tf.float32,[None, 16*15])
y=tf.placeholder(tf.float32,[None, 300])

x_image = tf.reshape(x,[-1,16,15,1])

W_conv1 = weight_variable([5,5,1,32]) # 5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_viaviable([32]) #每个卷积核一个偏置值
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)# 变成一个16*15*32
h_pool1 = max_pool_2x2(h_conv1)#变成8*8*32

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_viaviable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)#   8*8*64

###### 第二次池化之后变为 4*4*64
h_pool2 = max_pool_2x2(h_conv2)


# 第一个全连接层
W_fc1 = weight_variable([4*4*64,1024])
b_fc1 = bias_viaviable([1024])
# 4*4*64的图像变成1维向量
h_pool2_flat = tf.reshape(h_pool2,[-1,4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二个全连接层
W_fc2 = weight_variable([1024,300])
b_fc2 = bias_viaviable([300])
logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
prediction = tf.nn.relu(logits)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.003).minimize(loss)

prediction_2 = tf.nn.softmax(prediction)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = (tf.equal(tf.argmax(prediction_2,1), tf.argmax(y,1)))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

train_merged=tf.summary.merge_all()
test_merged=tf.summary.merge_all()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer_train=tf.summary.FileWriter('logs/train',sess.graph)
    writer_test = tf.summary.FileWriter('logs/test')
    Trainacc_list = []
    Testacc_list = []
    for global_step in range(200):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.get_next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        """
        输出训练集准确率
        """
        batch_xs, batch_ys = mnist.getTrain()
        #summary_train = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        summary_train, _train = sess.run([train_merged, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        writer_train.add_summary(summary_train, global_step)
        test_xs,test_ys=mnist.getTest()
        if global_step==100:
            pre = sess.run(prediction_2, feed_dict={x: test_xs, y: test_ys, keep_prob: 1.0})
        #summary_test = sess.run( accuracy, feed_dict={x: test_xs, y: test_ys, keep_prob: 1.0})
        summary_test,_test = sess.run([test_merged,accuracy], feed_dict={x:test_xs, y:test_ys, keep_prob:1.0})
        writer_test.add_summary(summary_test, global_step)
        print("Iter: " + str(global_step) + ", accTrain: " + str(_train)+", accTest: " + str(_test))

        Trainacc_list.append(100 * _train )
        Testacc_list.append(100 *_test)

    x1 = range(0, 200)
    x2 = range(0, 200)
    y1 = Trainacc_list
    y2 = Testacc_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.savefig("dynamic_accuracy_loss.png")

    saver.save(sess,'net/my_new.ckpt')

