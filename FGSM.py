from scipy.misc     import imsave
from keras          import metrics
from PIL            import Image
from keras.utils.np_utils import to_categorical
import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt
import sklearn.metrics

import os

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 用于 FGSM 算法的函数

# 获取梯度函数的符号表示
def get_gradient_signs(model, original_array):
    target_idx      = model.predict(original_array).argmax()
    target          = to_categorical(target_idx, 7)
    target_variable = K.variable(target)
    loss            = metrics.categorical_crossentropy(model.output, target_variable)
    gradients       = K.gradients(loss, model.input)
    get_grad_values = K.function([model.input], gradients)
    grad_values     = get_grad_values([original_array])[0]
    grad_signs      = np.sign(grad_values)
    
    return grad_signs


# 对图片进行扰动    
def pertubate_image(preprocessed_array, perturbation):
    modified_array  = preprocessed_array + perturbation
    deprocess_array = np.clip(modified_array, 0.0, 1.0)#.astype(np.uint8)   
    return deprocess_array

# 生成图片标题
def generate_titles(display_model, preprocessed_array, perturbation, modified_array):
    title_original     = generate_title(display_model, preprocessed_array)
    title_perturbation = generate_title(display_model, perturbation)
    title_modified     = generate_title(display_model, modified_array)   
    return title_original, title_perturbation, title_modified

# 生成对抗样本
def generate_adversarial_example(pertubation_model, original_array, epsilon):
    gradient_signs = get_gradient_signs(pertubation_model, original_array)
    perturbation   = gradient_signs * epsilon
    modified_image = pertubate_image(original_array, perturbation)  
    return modified_image, perturbation

# 读取图片
def load_image(filename):
    original_pic   = Image.open(filename).resize((48, 48))
    original_pic = np.array(original_pic) / 255

    pred_data = np.empty([1, 48, 48, 3])
    for index, item in enumerate(pred_data):
        item[:, :, 0] = original_pic
        item[:, :, 1] = original_pic
        item[:, :, 2] = original_pic
    return pred_data
    
def create_title(category, proba):
    return '"%s" %.1f%% confidence' % (category.replace('_', ' '), proba * 100) 

def generate_title(model, array):
    prediction = model.predict(array)
    categoryId = np.argmax(prediction)
    category = labels[categoryId]
    proba = prediction[0][categoryId]
    
    return create_title(category, proba)
    
# 生成对抗样本样例图片
def plot_adversarial_examples(folder, title, perturbation_model, display_model = None, epsilon = 0.01):
    if not display_model:
        display_model = perturbation_model

    filenames   = os.listdir(folder)
    line_number = len(filenames)
    plt.figure(figsize = (15, 10 * line_number))
    
    for line, filename in enumerate(filenames):
        original_array               = load_image(folder + filename)  
        modified_image, perturbation = generate_adversarial_example(perturbation_model, original_array, epsilon)
        orig_tit, pert_tit, modi_tit = generate_titles(display_model, original_array, perturbation, modified_image)

        plt.subplot(line_number, 3, 3 * line + 1)
        plt.imshow(original_array[0])
        plt.title(orig_tit)
        plt.subplot(line_number, 3, 3 * line + 3)
        plt.imshow(modified_image[0])
        plt.title(modi_tit)
        
    plt.suptitle(title)
    plt.tight_layout(pad = 4)


# 生成传入数据的对抗样本   
def generate_adversarial_examples(data_original, perturbation_model, epsilon = 0.01):
    data_adversarial = np.zeros((len(data_original), 48, 48, 3))
    print(data_adversarial.shape)
    i = 0
    for original_array in data_original:
        original_array = [original_array]
        original_array = np.array(original_array)
        modified_image, perturbation = generate_adversarial_example(perturbation_model, original_array, epsilon)
        data_adversarial[i] = modified_image
        if i % 10 == 0:
            print(i)
        i += 1
    return data_adversarial