import os
import shutil
import numpy as np
from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt
from collections import defaultdict
import timeit
from tkinter import messagebox

#internal functions
def ceildiv(a, b):
    return -(-a // b)

def isInt(value):
  try:
    int(value)
    return True
  except:
    return False

def get_image_shape(image_dictionary):
    return image_dictionary[list(image_dictionary)[0]].shape

def get_average_weight_matrix(image_dictionary, mean_vector):
    return np.reshape(mean_vector, get_image_shape(image_dictionary))

def recognize_unknown_face(image_dictionary_path, checking_image_dictionary):
    for s in list(checking_image_dictionary):
        if (os.path.isfile(os.path.join(image_dictionary_path, s))):
            os.remove(os.path.join(image_dictionary_path, s))

def get_Woptimal(pca_eigenvectors, lda_eigenvectors):
    return np.dot(pca_eigenvectors,lda_eigenvectors)

#main functions
def input_images(image_path, default_size):
    image_dictionary = defaultdict(list)
    image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
    onetime = True
    dimensions = 0
    for image_name in image_names:
        image = np.asarray(Image.open(os.path.join(image_path, image_name)))
        # print (image_name)
        # print (image)
        # print (image.shape)

        # dimensions = image.shape[0] * image.shape[1]

        # print (dimensions)
        downsample_factor1 = ceildiv(image.shape[0], default_size[0])
        downsample_factor2 = ceildiv(image.shape[1], default_size[1])
        downsample_factor = downsample_factor2
        if (downsample_factor1 >= downsample_factor2):
            downsample_factor = downsample_factor1

        if (downsample_factor>=2):
            image = skimage.transform.pyramid_reduce(image, downscale=downsample_factor)

        if (onetime==True):
            dimensions = image.shape[0] * image.shape[1]
            onetime=False

        if (len(image.shape) > 2):     
            image_dictionary[image_name] = skimage.color.rgb2gray(image)
        else:
            image_dictionary[image_name] = image
    return image_dictionary, dimensions 

def subjects_count(image_dictionary, stop_signal):
    subjects_names = list()
    for i in range(len(image_dictionary)):
        if (subjects_names.__contains__(list(image_dictionary)[i][0:list(image_dictionary)[i].find(stop_signal)]) == False):
            subjects_names.append(list(image_dictionary)[i][0:list(image_dictionary)[i].find(stop_signal)])
    return subjects_names

def get_total_vector_representation(image_dictionary, output_file = 'workfile.txt'):
    f = open(output_file, 'w')
    for ii, (image_name, image) in enumerate(image_dictionary.items()):
        if (ii == 0):
            # ds = list(image_dictionary)
            # print (ds)
            # vector_2d = image_dictionary[ds[0]].flatten()
            vector_2d = image.flatten()
        else:
            # if (ii == 650):
            #     print("Final")
            vector = image.flatten()
            f.writelines("Image: \n")
            f.writelines(str(image)+"\n")
            f.writelines("Image vector: \n")
            f.writelines(str(vector)+"\n")

            vector_2d = np.vstack((vector_2d, vector))
            f.writelines("vector 2d: \n"+ str(vector_2d)+"\n")
            # vector_2d = np.concatenate((vector_2d, vector), axis=0)
            # f.writelines("After cont: \n"+ str (vector_2d)+"\n" )
    # vector_2d = np.reshape(vector_2d, (len(image_dictionary), -1))
    # f.writelines("After reshape: \n"+ str (vector_2d))
    f.close()
    return vector_2d

def get_pca_eig(vector_matrix, k, ignore_first=0, output_file='pca_eig.txt'):
    f = open(output_file, 'w')
    # eigen_dict = defaultdict(list)
    mean_vector = vector_matrix.mean(axis=0)
    # print (mean_vector)

    # for ii in range(vector_matrix.shape[0]):
    #     vector_matrix[ii] = np.array(vector_matrix[ii] - mean_vector, dtype=np.float64)

    vector_matrix = np.array(vector_matrix - mean_vector, dtype=np.float64)

    # print (np.matmul(vector_matrix, vector_matrix.T))  

    eigen_values, eigen_vectors = np.linalg.eig(np.dot(vector_matrix, vector_matrix.T))
    eigen_vectors = np.dot(vector_matrix.T,eigen_vectors)
    # eigen_vectors, eigen_values, variances = np.linalg.svd(vector_matrix.T, full_matrices=False)
    # eigen_values = eigen_values ** 2 

    idx = np.argsort(-eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:,idx]

    eigen_values = np.array(eigen_values[ignore_first:k].real, dtype=np.float64, copy=True)
    eigen_vectors = np.matrix(eigen_vectors[0:,ignore_first:k].real, dtype=np.float64, copy=True)


    # for i in range(eigen_vectors.shape[0]):
    #     eigen_dict[tuple(eigen_vectors[i])] = eigen_values[i]

    f.writelines("Eigen_values: \n")
    f.writelines(str(eigen_values)+"\n")
    f.writelines("Eigen_vectors: \n")
    f.writelines(str(eigen_vectors)+"\n")

    f.close()

    # eigen_values_sorted_filtered = sorted(eigen_values, reverse=True)
    return mean_vector, eigen_vectors, eigen_values

def choose_best_eigen_vectors(eigen_values_sorted_filtered, eigen_dict, k, ignore_first_threes = False):
    best_eigen_vectors = []
    if (ignore_first_threes == True):
        for i in range(k):
            for eigen_vectors, eigen_values in eigen_dict.items():
                if (eigen_values == eigen_values_sorted_filtered[i] and i > 2):
                    best_eigen_vectors.append(eigen_vectors)
                    break

    else:
        for i in range(k):
            for eigen_vectors, eigen_values in eigen_dict.items():
                if (eigen_values == eigen_values_sorted_filtered[i]):
                    best_eigen_vectors.append(eigen_vectors)
                    break

    return np.array(best_eigen_vectors, dtype=np.float64)

def get_lda_eig(image_dictionary, stop_signal, num_components=3, output_file='lda_eig.txt'):
    f = open(output_file, 'w')
    shape = image_dictionary[list(image_dictionary)[0]].flatten().T.shape[0]
    Sw = np.zeros((shape, shape), dtype=np.float64)
    Sb = np.zeros((shape, shape), dtype=np.float64)
    mean_total = get_total_vector_representation(image_dictionary).mean(axis=0) 
    mean_total = np.reshape(mean_total, (-1,1))
    first_name = list(image_dictionary)[0][0:list(image_dictionary)[0].find(stop_signal)]
    first_number = 0
    for ii, (image_name, image) in enumerate(image_dictionary.items()):
        if (image_name[0:image_name.find(stop_signal)] == first_name):
            if (ii == 0):
                vector_2d = image.flatten().T
            else:
                vector_2d = np.concatenate((vector_2d, image.flatten().T), axis=0)
                    
        else:
            sample_number = ii-first_number
            vector_2d = np.reshape(vector_2d, (sample_number, -1))
            mean_class = vector_2d.mean(axis=0)
            for k in range(vector_2d.shape[0]):
                # print (np.array(vector_2d[k] - mean_class, dtype=np.float64))
                vector_2d[k] = np.array(vector_2d[k] - mean_class, dtype=np.float64)
                # vector_temp = np.reshape(vector_2d[k],(-1,1))
                # Sw += np.dot(vector_temp, vector_temp.T)
                # print (vector_2d[k])

            vector_2d = vector_2d.T

            Sw += np.dot(vector_2d, vector_2d.T)
            mean_class = np.reshape(mean_class, (-1,1))
            Sb += sample_number * np.dot((mean_class - mean_total), (mean_class - mean_total).T)

            first_name = list(image_dictionary)[ii][0:list(image_dictionary)[ii].find(stop_signal)]
            first_number = ii
            vector_2d = image.flatten().T

        if (ii >= len(image_dictionary)-1): 
            # print ("sas")
            sample_number = ii+1-first_number
            vector_2d = np.reshape(vector_2d, (sample_number, -1))
            mean_class = vector_2d.mean(axis=0)
            for k in range(vector_2d.shape[0]):
                # print (np.array(vector_2d[k] - mean_class, dtype=np.float64))
                vector_2d[k] = np.array(vector_2d[k] - mean_class, dtype=np.float64)
                # vector_temp = np.reshape(vector_2d[k],(-1,1))
                # Sw += np.dot(vector_temp, vector_temp.T)
                # print (vector_2d[k])

            vector_2d = vector_2d.T

            Sw += np.dot(vector_2d, vector_2d.T)
            mean_class = np.reshape(mean_class, (-1,1))
            Sb += sample_number * np.dot((mean_class - mean_total), (mean_class - mean_total).T)

    BinvA = np.dot(np.linalg.inv(Sw), Sb)
    eigenvalues, eigenvectors = np.linalg.eig(BinvA)

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float64, copy=True)
    eigenvectors = np.matrix(eigenvectors[0:,0:num_components].real, dtype=np.float64, copy=True)

    f.writelines("Sw: \n")
    f.writelines(str(Sw)+"\n")
    f.writelines("Sb: \n")
    f.writelines(str(Sb)+"\n")
    f.writelines("Eigen_values: \n")
    f.writelines(str(eigenvalues)+"\n")
    f.writelines("Eigen_vectors: \n")
    f.writelines(str(eigenvectors)+"\n")

    f.close()

    return eigenvalues, eigenvectors

def get_eigenface_space_coordination(image_dictionary, mean_vector, eigen_vectors, output_file='eigenface_positions.txt'):
    positions_dict = defaultdict(list)
    positions = list()

    f = open(output_file, 'a')

    for k in range(len(image_dictionary)):
        transform = np.reshape(np.array(image_dictionary[list(image_dictionary)[k]].flatten() - mean_vector, dtype=np.float64),(-1,1))
        result = np.dot(eigen_vectors.T, transform)
        # result = np.empty((eigen_vectors.shape[0],1), dtype=np.float64)
        # for v in range(eigen_vectors.shape[0]):
        #     result[v] = np.dot(eigen_vectors[v], transform)
        # print (result)

        f.writelines("------------------------------\n")
        f.writelines(str(list(image_dictionary)[k])+": \n"+str(result)+"\n")
       
        positions_dict[list(image_dictionary)[k]] = result
        positions.append(result)

    f.close()
    return positions, positions_dict

def get_fisherface_space_coordination(image_dictionary, eigen_vectors, output_file='fisherface_positions.txt'):
    positions_dict = defaultdict(list)
    positions = list()

    f = open(output_file, 'a')

    for k in range(len(image_dictionary)):
        transform = np.reshape(np.array(image_dictionary[list(image_dictionary)[k]].flatten(),np.float64),(-1,1))
        result = np.dot(eigen_vectors.T, transform)
        # print (result)

        f.writelines("------------------------------\n")
        f.writelines(str(list(image_dictionary)[k])+": \n"+str(result)+"\n")
       
        positions_dict[list(image_dictionary)[k]] = result
        positions.append(result)

    f.close()
    return positions, positions_dict

def get_min_distance_and_compare_threshold(positions_dict_check_eig, positions_data_eig, positions_dict_data_eig, threshold, rights, subjects_names, stop_signal, output_file='result.txt'):

    f = open(output_file, 'a')

    # Run only one time
    for checking_image_name, ckc in positions_dict_check_eig.items():
        Eud_distances = np.empty([len(positions_data_eig)], dtype=np.float64)
        # Run many times
        for y in range(len(positions_data_eig)):
            Eud_distances[y] = np.linalg.norm(np.subtract(ckc, positions_data_eig[y]))
            # print (Eud_distances[y])

        min_value_indice = np.argmin(Eud_distances)

        for image_name, kc in positions_dict_data_eig.items():
            if (((kc == positions_data_eig[min_value_indice]).all()) == True):

                f.writelines(str(checking_image_name) + ": \n" + str(ckc) + "\n")
                f.writelines("Nearest image in database: \n")
                f.writelines(str(image_name) + ": \n" + str(kc) + "\n") 
                f.writelines("With min distance: \n")
                f.writelines(str(Eud_distances[min_value_indice]) + "\n")

                for t in range(len(threshold)):
                    f.writelines("For threshold "+str(threshold[t])+":\n")
                    if (Eud_distances[min_value_indice] < threshold[t]):
                        f.writelines("--> " + str(image_name[0:image_name.find(stop_signal)]) +" - ")
                        if (image_name[0:image_name.find(stop_signal)] == checking_image_name[0:checking_image_name.find(stop_signal)]):
                            f.writelines("Right\n")
                            rights[t] += 1
                        else:
                            f.writelines("Wrong\n")
                    else:
                        f.writelines("--> Not recognized")
                        if (image_name[0:image_name.find(stop_signal)] == checking_image_name[0:checking_image_name.find(stop_signal)]):
                            f.writelines(" - Wrong\n")
                        else:
                            if (subjects_names.__contains__(checking_image_name[0:checking_image_name.find(stop_signal)])):
                                f.writelines(" - Wrong\n")
                            else:
                                f.writelines(" - Right\n")
                                rights[t] += 1

                f.writelines("------------------------------\n")
                break

    f.close()
    return rights

def get_mean_within_and_between_class_distance(positions_dict, stop_signal):
    ii = 0
    mean_within_subclass_distances = []
    mean_within_class_distances = []
    mean_between_subclass_distances = []
    count = 0
    reset = True
    init = True
    while (ii < len(positions_dict)):
        if (ii < len(positions_dict)-1):
            first_name = list(positions_dict)[ii][0:list(positions_dict)[ii].find(stop_signal)]
            total_distance = 0
            if (reset == True):
                mean_between_subclass_distances = positions_dict[list(positions_dict)[ii]]
                reset = False
            else:
                mean_between_subclass_distances = np.concatenate((mean_between_subclass_distances, positions_dict[list(positions_dict)[ii]]), axis=1)

            for i in range(ii+1,len(positions_dict)):
                if (list(positions_dict)[i][0:list(positions_dict)[i].find(stop_signal)] == first_name):              
                    count += 1
                    total_distance += np.linalg.norm(np.subtract(positions_dict[list(positions_dict)[ii]], positions_dict[list(positions_dict)[i]]))
                else:
                    if (ii == i-1):
                        if (count == 0):
                            count = 1
                        mean_within_class_distances = np.append(mean_within_class_distances, sum(mean_within_subclass_distances)/count)
                        mean_within_subclass_distances = []
                        if (init == True):
                            mean_class = np.reshape(mean_between_subclass_distances.mean(axis=1),(-1,1))
                            init = False
                        else:
                            mean_class = np.concatenate((mean_class, np.reshape((mean_between_subclass_distances.mean(axis=1)),(-1,1))),axis=1)
                        reset = True
                        count = 0

                    else:
                        mean_within_subclass_distances = np.append(mean_within_subclass_distances, total_distance)
                    ii += 1 
                    break
                if (i == len(positions_dict)-1):
                    mean_within_subclass_distances = np.append(mean_within_subclass_distances, total_distance)
                    if (ii == i-1):
                        mean_within_class_distances = np.append(mean_within_class_distances, sum(mean_within_subclass_distances)/count)
                        mean_between_subclass_distances = np.concatenate((mean_between_subclass_distances, positions_dict[list(positions_dict)[i]]), axis=1)
                        # mean_within_subclass_distances = []
                        if (init == True):
                            mean_class = np.reshape(mean_between_subclass_distances.mean(axis=1),(-1,1))
                            init = False
                        else:
                            mean_class = np.concatenate((mean_class, np.reshape((mean_between_subclass_distances.mean(axis=1)),(-1,1))),axis=1)
                        # count = 0
                        # reset = True
                        ii += 2
                    else:
                        ii += 1
        else:
            mean_within_class_distances = np.append(mean_within_class_distances, 0.)
            if (init == True):
                mean_class = positions_dict[list(positions_dict)[ii]]
            else:
                mean_class = np.concatenate((mean_class, positions_dict[list(positions_dict)[ii]]), axis=1)
            break
    mean_within_class_distance = mean_within_class_distances.mean(axis=0)
    if (mean_class.shape[1]==1):
        mean_between_class_distance = 0.
    else:
        x=0
        mean_between_class_distances = []
        while (x<mean_class.shape[1]-1):
            for y in range(x+1, mean_class.shape[1]):
                mean_between_class_distances = np.append(mean_between_class_distances, np.linalg.norm(np.subtract(mean_class[:,x], mean_class[:,y])))
            x +=1
        mean_between_class_distance = mean_between_class_distances.mean(axis=0)

    return mean_within_class_distance, mean_between_class_distance 

def plot_image_dictionary(image_dictionary):
    plt.gcf().clear()
    num_row_x = num_row_y = int(np.floor(np.sqrt(len(image_dictionary)-1))) + 1
    _, axarr = plt.subplots(num_row_x, num_row_y)
    for ii, (name, v) in enumerate(image_dictionary.items()):
        div, rem = divmod(ii, num_row_y)
        axarr[div, rem].imshow(v, cmap=plt.get_cmap('gray'))
        axarr[div, rem].set_title('{}'.format(name.split(".")[-1]).capitalize())
        axarr[div, rem].axis('off')
        if (ii == len(image_dictionary) - 1):
            for jj in range(ii, num_row_x*num_row_y):
                div, rem = divmod(jj, num_row_y)
                axarr[div, rem].axis('off')

def plot_mean_vector(image_dictionary, mean_vector):
    plt.gcf().clear()
    plt.imshow(get_average_weight_matrix(image_dictionary, mean_vector), cmap=plt.get_cmap('gray'))
    plt.show()

def plot_eigen_vector(image_dictionary, eigen_vectors, n_eigen=0):
    plt.gcf().clear()
    plt.imshow(np.array(np.reshape(eigen_vectors[:,n_eigen], get_image_shape(image_dictionary)),dtype=np.float64), cmap=plt.get_cmap('gray'))
    plt.show()

def plot_eigen_vectors(image_dictionary, eigen_vectors):
    # number = 10
    number = eigen_vectors.shape[1]
    num_row_x = num_row_y = int(np.floor(np.sqrt(number-1))) + 1
    _, axarr = plt.subplots(num_row_x, num_row_y)
    for ii in range(number):
        div, rem = divmod(ii, num_row_y)
        axarr[div, rem].imshow(np.array(np.reshape(eigen_vectors[:,ii], get_image_shape(image_dictionary)),dtype=np.float64), cmap=plt.get_cmap('gray'))
        axarr[div, rem].axis('off')
        if (ii == number - 1):
            for jj in range(ii, num_row_x*num_row_y):
                div, rem = divmod(jj, num_row_y)
                axarr[div, rem].axis('off')
    plt.show()

def plot_eigen_values(eigen_values_sorted_filtered):
    # plt.xticks(np.arange(0, len(eigen_values_sorted_filtered), 1))
    # plt.yticks(np.arange(0, eigen_values_sorted_filtered[len(eigen_values_sorted_filtered)-1], 100))
    plt.gcf().clear()
    plt.plot(np.arange(len(eigen_values_sorted_filtered)), eigen_values_sorted_filtered, 'ro')

def plot_scatter(positions_dict, title, stop_signal):
    markers = ['.',	',', 'o', 'v', '^',	'<', '>', '1', '2',	'3', '4', '8', 's',	'p', '*', 'h', 'H',	'+', 'x', 'D', 'd', '|', '_']
    markers_reduced = ['.',	',', 'o']
    i = 0
    first_name = list(positions_dict)[0][0:list(positions_dict)[0].find(stop_signal)]
    x = []
    y = []
    plt.gcf().clear()

    for ii, (image_name, position) in enumerate(positions_dict.items()):
        # chosen = markers_reduced
        chosen = markers
        if (i<len(chosen)):    
            if (image_name[0:image_name.find(stop_signal)] == first_name):
                x = np.append(x, position[0])
                y = np.append(y, position[1])

            else:
                plt.scatter(x, y, marker=chosen[i], label=first_name)
                plt.legend(scatterpoints=1, loc='best', fontsize=8, ncol=2)
                first_name = list(positions_dict)[ii][0:list(positions_dict)[ii].find(stop_signal)]
                i += 1
                x = np.append([], position[0])
                y = np.append([], position[1])

            if (ii >= len(positions_dict)-1):
                plt.scatter(x, y, marker=chosen[i], label=first_name)
                plt.legend(scatterpoints=1, loc='best', fontsize=8, ncol=2)

        else:
            break

    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title(title)

def plot_scatter_UI(positions_dict, title, stop_signal):
    plot_scatter(positions_dict, title, stop_signal)
    plt.show()

def run_eigenface(image_dir, stop_signal, check_image_dir, default_size, threshold, k=-1, clear=True, copy=True, ignore_first = 0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt'):
    image_dictionary_path = image_dir
    checking_image_dictionary_path = check_image_dir
    default_size = default_size
    image_dictionary = defaultdict(list)

    image_dictionary, dimensions_data = input_images(image_dictionary_path, default_size)
    checking_image_dictionary, dimensions_check = input_images(checking_image_dictionary_path, default_size)
    if (len(checking_image_dictionary) > 0 and len(checking_image_dictionary) > 0 and dimensions_data == dimensions_check):
        if (clear == True):
            recognize_unknown_face(image_dictionary_path, checking_image_dictionary)

        image_dictionary, dimensions = input_images(image_dictionary_path, default_size)
        rights = np.zeros(len(threshold))

        # plot_image_dictionary(image_dictionary)
        # plt.show()
        # plot_image_dictionary(checking_image_dictionary)
        # plt.show()

        if (k==-1):
            k = input("Enter number of desired eigenfaces (<=" + str(len(image_dictionary)) + "): ")
        elif (k==-2):
            k = len(image_dictionary)
            print ("Use default value of k ("+str(len(image_dictionary))+")")

        if (isInt(k) and int(k) > 0 and int(k) <= len(image_dictionary)):
            k = int(k)

            #Preparing function
            open(results, 'w').close()
            open(positions_list, 'w').close()

            if (copy==False):
                subjects_names = subjects_count(image_dictionary, stop_signal)

                vector_matrix = get_total_vector_representation(image_dictionary)

                mean_vector, eigen_vectors_eig, eigen_values = get_pca_eig(vector_matrix, k, ignore_first)

                # best_eigen_vectors= choose_best_eigen_vectors(eigen_values_sorted_filtered, eigen_dict, k, ignore_first_threes)

                positions_data_eig, positions_dict_data_eig = get_eigenface_space_coordination(image_dictionary, mean_vector, eigen_vectors_eig)

            
            for x in range(len(checking_image_dictionary)):
                temp_image_dictionary = defaultdict(list)
                temp_image_dictionary[list(checking_image_dictionary)[x]] = checking_image_dictionary[list(checking_image_dictionary)[x]]

                if (copy==True):
                    # image_dictionary = input_images(image_dictionary_path, default_size)
                    subjects_names = subjects_count(image_dictionary, stop_signal)

                    vector_matrix = get_total_vector_representation(image_dictionary)

                    mean_vector, eigen_vectors_eig, eigen_values = get_pca_eig(vector_matrix, k, ignore_first)
                    # mean_vector, eigen_vectors_svd, singular_values, eigen_values = get_pca_svd(vector_matrix)

                    # plot_mean_vector(image_dictionary, mean_vector)
                    # plt.show()

                    # plot_eigen_vector(image_dictionary, eigen_vectors_eig)
                    # plt.show()

                    # plot_eigen_vectors(image_dictionary, eigen_vectors_eig)
                    # plt.show()
                    # plot_eigen_vectors(image_dictionary, eigen_vectors_svd)
                    # plt.show()

                    # plot_eigen_values(eigen_values_sorted_filtered)
                    # plt.show()

                    # best_eigen_vectors= choose_best_eigen_vectors(eigen_values_sorted_filtered, eigen_dict, k, ignore_first_threes)

                    # plot_eigen_vectors(image_dictionary, best_eigen_vectors)
                    # plt.show()

                    positions_data_eig, positions_dict_data_eig = get_eigenface_space_coordination(image_dictionary, mean_vector, eigen_vectors_eig)

                positions_check_eig, positions_dict_check_eig = get_eigenface_space_coordination(temp_image_dictionary, mean_vector, eigen_vectors_eig)

                rights = get_min_distance_and_compare_threshold(positions_dict_check_eig, positions_data_eig, positions_dict_data_eig, threshold, rights, subjects_names, stop_signal, results)
                
                if (copy == True):
                    shutil.copy2(os.path.join(checking_image_dictionary_path, list(temp_image_dictionary)[0]),image_dictionary_path)
                    image_dictionary[list(checking_image_dictionary)[x]] = checking_image_dictionary[list(checking_image_dictionary)[x]]
                    print ("Learned" + list(temp_image_dictionary)[0])
                # else:
                #     print ("Done recognizing " + list(temp_image_dictionary)[0])
                
                if (x+1==len(checking_image_dictionary)):
                    plot_scatter(positions_dict_data_eig, title, stop_signal)
                    # plt.show()
                    mean_within_class_distance, mean_between_class_distance = get_mean_within_and_between_class_distance(positions_dict_data_eig, stop_signal)
                    print ("---------------")
                    print ("Mean within class distance: " + str(mean_within_class_distance))
                    print ("Mean between class distance: " + str(mean_between_class_distance))
                    print ("Within/Between ratio: " + str(mean_within_class_distance/mean_between_class_distance))
                # input('--> ')
            
            # print (check_image_dir)
            print ("K: " + str(k))
            for t in range(len(threshold)): 
                print ("Threshold: "+ str(threshold[t]))
                print ("Right: " + str(rights[t]) + "/" + str(len(checking_image_dictionary)))
                success_rate = np.dot(rights[t]/(len(checking_image_dictionary)), 100)
                print ("Success rate: "+ str(success_rate) + "%")
                print ("Result was stored in "+str(os.path.abspath(results)))
                print ("Detailed positions were stored in "+str(os.path.abspath(positions_list)))
                print ("---------------")
            
            print ("Done!")
            return k, threshold, rights[t], len(checking_image_dictionary), success_rate, os.path.abspath(results), os.path.abspath(positions_list), mean_within_class_distance, mean_between_class_distance, positions_dict_data_eig, image_dictionary, mean_vector, eigen_vectors_eig

        else:
            messagebox.showinfo("Error","Please enter a valid positive integer(<="+ str(len(image_dictionary)) + ")!")
            # print ("Failed!")
    elif (ignore_first>=k):
        messagebox.showinfo("Error","Invalid ignore_first or k parameter!")
        # print ("Failed!")
    else:
        messagebox.showinfo("Error","Invalid database or checking image(s)!")
        # print ("Failed!")

#Most done!
def run_fisherface(image_dir, stop_signal, check_image_dir, default_size, threshold, k=-1, m=-1, clear=True, copy=True, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt'):
    image_dictionary_path = image_dir
    checking_image_dictionary_path = check_image_dir
    default_size = default_size
    image_dictionary = defaultdict(list)

    image_dictionary, dimensions_data = input_images(image_dictionary_path, default_size)
    checking_image_dictionary, dimensions_check = input_images(checking_image_dictionary_path, default_size)
    if (len(checking_image_dictionary) > 0 and len(image_dictionary) > 0 and dimensions_data==dimensions_check):
        if (clear == True):    
            recognize_unknown_face(image_dictionary_path, checking_image_dictionary)

        image_dictionary, dimensions = input_images(image_dictionary_path, default_size)
        rights = np.zeros(len(threshold))
        # plot_image_dictionary(image_dictionary)
        # plt.show()
        # plot_image_dictionary(checking_image_dictionary)
        # plt.show()
        
        #check general condition of k
        if (k==-1):
            k = input("Enter number of desired eigenfaces (<" + str(len(image_dictionary)) + "): ")

        if (isInt(k) and (int(k) > 0 or int(k) == -2) and int(k) < len(image_dictionary)):
            k = int(k)

            #Preparing function
            open(results, 'w').close()
            open(positions_list, 'w').close()

            if (copy==False):
                subjects_names = subjects_count(image_dictionary, stop_signal)
                #check specific condition of k
                default_k = len(image_dictionary)-len(subjects_names)
                if (k==-2 or k > default_k):
                    k = default_k
                    print ("Use default value of k ("+str(k)+")")

                #check input m
                if (m==-1):
                    m = input("Enter number of desired fisherfaces (<=" + str(len(subjects_names)-1) + "): ")
                elif (m==-2):
                    m = len(subjects_names)-1
                    print ("Use default value of m ("+str(len(subjects_names)-1)+")")
                while (isInt(m)==False or int(m) < 0 or int(m) > len(subjects_names)-1):
                    m = input("Enter number of desired fisherfaces (<=" + str(len(subjects_names)-1) + "): ")

                vector_matrix = get_total_vector_representation(image_dictionary)

                mean_vector, eigen_vectors_eig, eigen_values = get_pca_eig(vector_matrix, k)

                #row matrix
                # pca_best_eigen_vectors = choose_best_eigen_vectors(eigen_values_sorted_filtered, eigen_dict, k) 

                positions_data_eig, positions_dict_data_eig = get_eigenface_space_coordination(image_dictionary, mean_vector, eigen_vectors_eig)
                
                #column matrix
                lda_eigenvalues, lda_eigenvectors = get_lda_eig(positions_dict_data_eig, stop_signal, num_components=m) 

                optimal_eigenvectors = get_Woptimal(eigen_vectors_eig, lda_eigenvectors)

                positions_final_data, positions_final_dict_data = get_fisherface_space_coordination(image_dictionary, optimal_eigenvectors)

            ini = True

            for x in range(len(checking_image_dictionary)):
                temp_image_dictionary = defaultdict(list)
                temp_image_dictionary[list(checking_image_dictionary)[x]] = checking_image_dictionary[list(checking_image_dictionary)[x]]

                if (copy==True):
                    # image_dictionary = input_images(image_dictionary_path, default_size)
                    subjects_names = subjects_count(image_dictionary, stop_signal)

                    if (ini==True):
                        #check specific condition of k
                        default_k = len(image_dictionary)-len(subjects_names)
                        if (k==-2 or k > default_k):
                            k = default_k
                            print ("Use default value of k ("+str(k)+")")

                        #check input m
                        if (m==-1):
                            m = input("Enter number of desired fisherfaces (<=" + str(len(subjects_names)-1) + "): ")
                        elif (m==-2):
                            m = len(subjects_names)-1
                            print ("Use default value of m ("+str(len(subjects_names)-1)+")")
                        while (isInt(m)==False or int(m) < 0 or int(m) > len(subjects_names)-1):
                            m = input("Enter number of desired fisherfaces (<=" + str(len(subjects_names)-1) + "): ")
                        ini=False

                    vector_matrix = get_total_vector_representation(image_dictionary)

                    mean_vector, eigen_vectors_eig, eigen_values = get_pca_eig(vector_matrix, k)
                    # mean_vector, eigen_vectors_svd, singular_values, eigen_values = get_pca_svd(vector_matrix)

                    # plot_mean_vector(image_dictionary, mean_vector)
                    # plt.show()

                    # plot_eigen_vector(image_dictionary, eigen_vectors_eig)
                    # plt.show()

                    # plot_eigen_vectors(image_dictionary, eigen_vectors_eig)
                    # plt.show()
                    # plot_eigen_vectors(image_dictionary, eigen_vectors_svd)
                    # plt.show()

                    # plot_eigen_values(eigen_values_sorted_filtered)
                    # plt.show()

                    #row matrix
                    # pca_best_eigen_vectors = choose_best_eigen_vectors(eigen_values_sorted_filtered, eigen_dict, k)

                    # plot_eigen_vectors(image_dictionary, pca_best_eigen_vectors)
                    # plt.show()

                    positions_data_eig, positions_dict_data_eig = get_eigenface_space_coordination(image_dictionary, mean_vector, eigen_vectors_eig)
                    
                    #column matrix
                    lda_eigenvalues, lda_eigenvectors = get_lda_eig(positions_dict_data_eig, stop_signal, num_components=m)

                    optimal_eigenvectors = get_Woptimal(eigen_vectors_eig, lda_eigenvectors)

                    positions_final_data, positions_final_dict_data = get_fisherface_space_coordination(image_dictionary, optimal_eigenvectors)

                positions_check, positions_dict_check = get_fisherface_space_coordination(temp_image_dictionary, optimal_eigenvectors)

                rights = get_min_distance_and_compare_threshold(positions_dict_check, positions_final_data, positions_final_dict_data, threshold, rights, subjects_names, stop_signal, results)
                
                if (copy == True):
                    shutil.copy2(os.path.join(checking_image_dictionary_path, list(temp_image_dictionary)[0]),image_dictionary_path)
                    image_dictionary[list(checking_image_dictionary)[x]] = checking_image_dictionary[list(checking_image_dictionary)[x]]
                    print ("Learned " + list(temp_image_dictionary)[0])
                # else:
                #     print ("Done recognizing " + list(temp_image_dictionary)[0])

                if (x+1==len(checking_image_dictionary)):
                    plot_scatter(positions_final_dict_data, title, stop_signal)
                    # plt.show()
                    mean_within_class_distance, mean_between_class_distance = get_mean_within_and_between_class_distance(positions_final_dict_data, stop_signal)
                    print ("---------------")
                    print ("Mean within class distance: " + str(mean_within_class_distance))
                    print ("Mean between class distance: " + str(mean_between_class_distance))
                    print ("Within/Between ratio: " + str(mean_within_class_distance/mean_between_class_distance))
                # input('--> ')
            
            # print (check_image_dir)
            print ("K: " + str(k))
            for t in range(len(threshold)): 
                print ("Threshold: "+ str(threshold[t]))
                print ("Right: " + str(rights[t]) + "/" + str(len(checking_image_dictionary)))
                success_rate = np.dot(rights[t]/(len(checking_image_dictionary)), 100)
                print ("Success rate: "+ str(success_rate) + "%")
                print ("Results were stored in "+str(os.path.abspath(results)))
                print ("Detailed positions were stored in "+str(os.path.abspath(positions_list)))
                print ("---------------")

            print ("Done!")
            return k, m, threshold, rights[t], len(checking_image_dictionary), success_rate, os.path.abspath(results), os.path.abspath(positions_list), mean_within_class_distance, mean_between_class_distance, positions_final_dict_data, image_dictionary, mean_vector, optimal_eigenvectors

        else:
            messagebox.showinfo("Error", "Please enter a valid positive integer(<"+ str(len(image_dictionary)) + ")!")
    else:
        messagebox.showinfo("Error", "Please input at least an image!")

#YALEFACE A normal
# run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking2'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# f1 = plt.figure(1)
# print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking'), [128, 128], threshold=[500], k=40, clear=True, copy=False, ignore_first=0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking2'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=2, title='EigenFace_ignore first threes', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# f2 = plt.figure(2)
# print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking'), [128, 128], threshold=[500], k=40, clear=True, copy=False, ignore_first=3, title='EigenFace_ignore first threes', results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# run_fisherface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking2'), [128, 128], threshold=[500], k=40, m=-2, clear=True, copy=True, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt')
# f3 = plt.figure(3)
# print(timeit.timeit("run_fisherface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/normal/random_checking'), [128, 128], threshold=[500], k=40, m=-2, clear=True, copy=False, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt')", globals=globals(), number=1))

# plt.show()

#YALEFACE A difficult
# run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/glasses and light'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# f1 = plt.figure(1)
# print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/light'), [128, 128], threshold=[500], k=40, clear=True, copy=False, ignore_first=0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# # run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/glasses and light'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=3, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# # f2 = plt.figure(2)
# # print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces2.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/light2'), [128, 128], threshold=[500], k=-2, clear=True, copy=False, ignore_first=3, results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# # run_fisherface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/glasses and light'), [128, 128], threshold=[500], k=40, m=-2, clear=True, copy=True, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt')
# f3 = plt.figure(3)
# print(timeit.timeit("run_fisherface(os.path.abspath('EigenFace/database/YaleFaceA/old_faces/training_set/distinct_yalefaces.jpg'), '.', os.path.abspath('EigenFace/database/YaleFaceA/old_faces/testing_set/difficult/light'), [128, 128], threshold=[500], k=40, m=-2, clear=True, copy=False, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt')", globals=globals(), number=1))

# plt.show()

#YALEFACE B
# run_eigenface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/CroppedYale.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=0, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# f1 = plt.figure(1)
# print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/reduced_croppedyaleB.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[10000], k=40, clear=True, copy=False, ignore_first=0, results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# run_eigenface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/CroppedYale.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[500], k=40, clear=True, copy=True, ignore_first=3, title='EigenFace', results='eigenface_results.txt', positions_list='eigenface_positions.txt')
# f2 = plt.figure(2)
# print(timeit.timeit("run_eigenface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/reduced_croppedyaleB.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[500], k=40, clear=True, copy=False, ignore_first=3, results='eigenface_results.txt', positions_list='eigenface_positions.txt')", globals=globals(), number=1))

# run_fisherface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/CroppedYale.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[500], k=40, m=-2, clear=True, copy=True, title='FisherFace', results='fisherface_results.txt', positions_list='fisherface_positions.txt')
# f3 = plt.figure(3)
# print(timeit.timeit("run_fisherface(os.path.abspath('EigenFace/database/YaleFaceB/old_faces/training_set/reduced_croppedyaleB.jpg'), '_', os.path.abspath('EigenFace/database/YaleFaceB/old_faces/testing_set/normal'), [128, 128], threshold=[10000], k=40, m=-2, clear=True, copy=False, results='fisherface_results.txt', positions_list='fisherface_positions.txt')", globals=globals(), number=1))

# plt.show()

