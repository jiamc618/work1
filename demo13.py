import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt     #图像的显示
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from time import time
tf.disable_eager_execution()

#读取mnist数据
mnist = input_data.read_data_sets(r"D:\编程代码\python程序\mnist_data",one_hot = True)   #r表示原始字符串


#构建输入层
x = tf.placeholder(tf.float32,[None,784],name="X")   #每张图片28*28个像素点
y = tf.placeholder(tf.float32,[None,10],name="Y")

#构建隐藏层
H1_NN = 256     #第1隐藏层 神经元为256个
H2_NN = 64     #第2隐藏层神经元为64个
H3_NN = 32    #第3隐藏层神经元为32个

#定义全连接层函数
def fcn_layer(inputs,input_dim,output_dim,activation=None):
    """
    inputs:输入数据
    input_dim:输入神经元数量
    output_dim:输出神经元数量
    activation:激活函数
    
    """
    W = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))
    
    XWb = tf.matmul(inputs,W) + b   #矩阵相乘
    
    outputs = XWb if (activation is None) else activation(XWb) 
    
    return outputs
        
h1 = fcn_layer(inputs=x,input_dim=784,output_dim=H1_NN,activation=tf.nn.relu)
h2 = fcn_layer(inputs=h1,input_dim=H1_NN,output_dim=H2_NN,activation=tf.nn.relu)
h3 = fcn_layer(inputs=h2,input_dim=H2_NN,output_dim=H3_NN,activation=tf.nn.relu)



forward =  fcn_layer(inputs=h3,input_dim=H3_NN,output_dim=10,activation=None)
pred = tf.nn.softmax(forward)   #Softmax 分类


#设置训练参数
train_epochs = 40  #训练轮数
batch_size = 50   #单次训练样本数
total_batch = int(mnist.train.num_examples/batch_size)   #一轮训练有多少批次
display_step = 1   #显示粒度
learning_rate = 0.008  #学习率


#定义损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))

#选择AdamOptimizer优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)


#定义准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))  #argmax将最大标签取出来

#准确率,将布尔型转换为浮点型，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


startTime = time()
sess = tf.Session()  #声明回话
init = tf.global_variables_initializer()  #变量的初始化
sess.run(init)

#开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys = mnist.train.next_batch(batch_size)  #读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})   #执行批次训练
        
        
    #使用验证集计算误差和准确率
    loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    
    #打印训练中的详细信息
    if (epoch+1) % display_step == 0:
        print("Train Epoch:{:02d} Loss={:.9f} Accuracy={:.4f}".format(epoch+1,loss,acc))
    
print("Train Finished！")
#显示运行时间
duration = time() - startTime
print("Train Finished takes:{:.2f}".format(duration))
print("\n\n")


#完成训练后,在测试集上评估模型的准确率
accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy：",accu_test)

#完成训练后,在验证集上评估模型的准确率
accu_validation = sess.run(accuracy,feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
print("Validation Accuracy：",accu_validation)


#完成训练后,在测试集上评估模型的准确率
accu_train = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels})
print("Train Accuracy：",accu_train)
print("\n\n")


#预测结果
prediction_result = sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})

#找出预测错误
compare_lists = prediction_result==np.argmax(mnist.test.labels,1)
#列表推导式
err_list = [i for i in range(len(compare_lists)) if compare_lists[i] ==False]


#定义可视化函数
def plot_images_labels_prediction(images,labels,prediction,num=10):
    #函数文档
    """
    images:图像列表
    labels:标签列表
    prediction:预测值列表
    index:从第index个开始显示
    num=10：缺省依次显示10幅
    
    """
    j=-1 
    fig = plt.gcf()    #获取当前图标
    fig.set_size_inches(10,12)
    if num > 25:
        num = 25  #最多显示25个子图
    for i in  range(num):
        j += 1
        index = err_list[j]
        ax = plt.subplot(5,5,i+1)     #获取当前需要处理的子图
        #显示第index个图像
        ax.imshow(np.reshape(images[index],(28,28)),cmap="binary")
        title =  "label=" + str(np.argmax(labels[index]))  #标题
        if len(prediction) > 0:
            title += (",predict=" + str(prediction[index]))
            
        ax.set_title(title,fontsize=10)    #显示标题信息字体大小10号
        ax.set_xticks([])   #不显示坐标轴
        ax.set_yticks([])
    plt.show()


print("可视化查看错误的样本：")
#显示15幅错误预测值与标签值对比图
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,15)  
