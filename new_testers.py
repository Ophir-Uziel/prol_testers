import os
import face_recognition
import time
import shutil

import FN_functions as FN
import dnn.face_detection_opencv_dnn as dnn
from sqlalchemy.dialects.mssql import SMALLDATETIME

from new_functions import *
from matplotlib.pyplot import *
import random
import resize

DEBUG = False
DATA_SET_FOLDER_PATH = "/home/ophir/Desktop/dataSets/lfw"
BIG_DS = '/mnt/d/Prol/data/general_ds/'
SMALL_DS = '/mnt/d/Prol/data/small_ds/'
TRY_DS = 'imgs/'
ALGORITHM = 'FR'
RESOLUTIONS = [(1800, 1800)]
num_images_load = 30
save = False


def print_parameters(resolution, true_positive, true_negative, false_positive, false_negative,
                     avg_compare_time_on_others, avg_compare_time_on_himself):
    print(" RESOLUTION : ", resolution)
    print("\ttrue_positive : ", true_positive, " | true_negative : ", true_negative, " | false_positive : ",
          false_positive,
          " | false_negative : ", false_negative, ". Running Tme - on others: ", avg_compare_time_on_others,
          "| on himself : ", avg_compare_time_on_himself)

    f = open("results.txt", "w+")
    f.write(" RESOLUTION : " + str(resolution) + "\n")
    f.write("\ttrue_positive : " + str(true_positive) + " | true_negative : " + str(
        true_negative) + " | false_positive : " +
            str(false_positive) +
            " | false_negative : " + str(false_negative) + ". Running Tme - on others: " + str(
        avg_compare_time_on_others) +
            "| on himself : " + str(avg_compare_time_on_himself) + "\n")
    f.close()


# create a folder if not exist
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# prints if DEBUG == TRUE
def debug_print(message):
    if DEBUG:
        print(message)


#upload data from files
def load_images(main_dir_path, start_index):
    # region : Get all the pictures:
    # in format : [image, true if the person has a few images, false if only 1]
    images = []
    for subdir, dirs, files in os.walk(main_dir_path):
        for dir in dirs:
            if len(images) >= num_images_load:
                continue

            pic_dir_path = main_dir_path + "/" + dir
            images_path_list_in_dir = os.listdir(pic_dir_path)
            should_add = False
            new_person_images = []
            for i in range(len(images_path_list_in_dir)):
                image = images_path_list_in_dir[i]
                image_path = pic_dir_path + "/" + image
                try:
                    new_person_images.append(face_recognition.load_image_file(image_path))
                    if image == 'profile.jpg':
                        should_add = True
                except:
                    debug_print("image" + str(i) + "is bad form")
                    if image == 'profile.jpg':
                        should_add = False
                        continue
            if should_add:
                images.append([new_person_images, dir])
            else:
                shutil.rmtree(pic_dir_path)
                debug_print(dir + " dir was deleted")
    return images
    # end region


# get person from all person, his images and name
def get_person_from_images(images, person_index):
    person = images[person_index]
    person_images = person[0]
    person_name = person[1]
    return person, person_images, person_name


# save 2 images of same person
def save_2_images_same_person(path, person_name, image1, image2, index1, index2):
    current_comparison_results_folder = path + person_name + '||' + str(index1) + str(index2) + '/'
    create_folder(current_comparison_results_folder)

    if save:
        imsave(current_comparison_results_folder + str(index1), image1)
        imsave(current_comparison_results_folder + str(index2), image2)


def save_image_in_path(path, person_name, i, image):
    current_comparison_results_folder = path + person_name + '/'
    create_folder(current_comparison_results_folder)
    if save:
        imsave(current_comparison_results_folder + i, image)


def save_faces_in_path(path, person_name, j, faces):
    faces_folder = path + person_name + '/' + j + 'faces/'
    create_folder(faces_folder)
    if save:
        for i in range(len(faces)):
            face = faces[i]
            if ALGORITHM == 'FN':
                face = face['face']
            imsave(faces_folder + str(i), face)

def save_faces_2_person(path,face1, face2, name1, name2):
    path1 = path + name1 + "||" + name2 + '/face | ' + name1
    path2 = path + name1 + "||" + name2 + '/face | ' + name2
    if save:
        if ALGORITHM == 'FN':
            face1 = face1['face']
            face2 = face2['face']
        imsave(path1, face1)
        imsave(path2, face2)

# save 2 images of different persons
def save_2_images_different_persons(path, image1, image2, name1, name2):
    current_comparison_results_folder = path + name1 + "||" + name2 + '/'
    create_folder(current_comparison_results_folder)

    if save:
        imsave(current_comparison_results_folder + name1, image1)
        imsave(current_comparison_results_folder + name2, image2)


# takes person's images and checks if the algorithm says it is the same person
def compare_person_to_himself(person_images, person_name, compared_himself_path, not_exist_path):
    num_of_images = len(person_images)

    debug_print(person_name + str(num_of_images))

    for i in range(num_of_images):

        # check the picture on the same person's pictures
        for j in range(i + 1, num_of_images):
            current_image = person_images[i]
            check_image = person_images[j]

            start_time = time.time()

            result,m,n = compare_images(current_image, check_image, ALGORITHM)
            debug_print("comparing time = " + str(time.time() - start_time))

            if result == DIFFERENT:
                debug_print("Ooh, it just said a person isn't the same with himself")
                save_2_images_same_person(compared_himself_path, person_name, current_image, check_image, i, j)

            elif result == SAME:
                debug_print("Great! found the person identical to itself!")

            elif result == FIRST_IMAGE_NO_FACES and save:
                imsave(not_exist_path + person_name + str(i), current_image)
            elif save:
                imsave(not_exist_path + person_name + str(j), check_image)


def get_profile(person_images):
    # the last image in a folder is always the profile
    profile = person_images[len(person_images) - 1]
    return profile


def get_k_pictures_num_to_check(k, person_images):
    # NOTE : here it is assuming that the profile image is the last in the list!
    n = len(person_images)
    if n < k + 1:
        return set(range(n))

    nums = set()

    choose_from = [i for i in range(n - 1)]

    while len(nums) < k:
        new_check_index = random.choice(choose_from)
        choose_from.remove(new_check_index)
        nums.add(new_check_index)

    return nums


def get_random_indexes(person_index, num_of_people, num_of_checks):
    nums = set()

    choose_from = [i for i in range(num_of_people)]
    choose_from.remove(person_index)

    while len(nums) < num_of_checks:
        new_check_index = random.choice(choose_from)
        choose_from.remove(new_check_index)
        nums.add(new_check_index)

    return nums


def get_avg_compare_time(num_checks_avg_time_person, num_of_checks):
    avg_time = 0
    for i in range(int(len(num_checks_avg_time_person) / 2)):
        avg_time += num_checks_avg_time_person[i] * num_checks_avg_time_person[i + 1] / num_of_checks
    return avg_time


def compare_person_to_himself_up_to_k(k, person_images, person_name, compared_himself_path, not_exist_path, resolution):
    num_of_images = len(person_images)

    debug_print(person_name + str(num_of_images))

    profile_image = get_profile(person_images)

    profile_image = resize.resize(profile_image, resolution)

    pictures_nums_to_check = get_k_pictures_num_to_check(k, person_images)

    checks_made = 0

    profile_faces = dnn.get_faces_from_image(profile_image)
    # profile_loc_faces = get_faces_loc(profile_image, ALGORITHM)
    # profile_faces = get_faces(profile_image, ALGORITHM, profile_loc_faces)
    add_profile = True

    avg_time = 0
    for i in pictures_nums_to_check:
        try:
            check_image = person_images[i]
        except:
            debug_print("")

        start_time = time.time()
        check_image = resize.resize(check_image, resolution)
        # check_image_loc_faces = get_faces_loc(check_image, ALGORITHM)
        # check_image_faces = get_faces(check_image, ALGORITHM, check_image_loc_faces)
        check_image_faces = dnn.get_faces_from_image(check_image)
        result,m,n = FN.compare_faces_FN(profile_faces, check_image_faces)
        # result,m,n = compare_images(profile_image, check_image, ALGORITHM, face_locs1=profile_loc_faces,
        #                         face_locs2=check_image_loc_faces, faces1=profile_faces, faces2=check_image_faces)

        checks_made += 1

        avg_time += time.time() - start_time
        debug_print("comparing time = " + str(time.time() - start_time))

        if result == DIFFERENT:
            debug_print("Ooh, it just said a person isn't the same with himself")
            save_image_in_path(compared_himself_path, person_name, str(i), check_image)
            save_faces_in_path(compared_himself_path, person_name, str(i), check_image_faces)
            if add_profile:
                save_image_in_path(compared_himself_path, person_name, 'profile', profile_image)
                save_faces_in_path(compared_himself_path, person_name, 'profile', profile_faces)
                add_profile = False


        elif result == SAME:
            debug_print("Great! found the person identical to itself!")
            avg_time /= checks_made
            debug_print("avg_time = " + str(avg_time))

            return checks_made, True, avg_time

        else:
            imsave(not_exist_path + person_name + str(i), check_image)

    avg_time /= checks_made

    debug_print("avg_time = " + str(avg_time))

    return checks_made, False, avg_time


# create all dirs needed for result images
def create_results_dir(results_folder, compared_himself_folder, failed_folder, not_exist_folder):
    failed_path = results_folder + failed_folder
    not_exist_path = results_folder + not_exist_folder
    compared_himself_path = results_folder + compared_himself_folder

    create_folder(results_folder)
    create_folder(compared_himself_path)
    create_folder(failed_path)
    create_folder(not_exist_path)

    return compared_himself_path, failed_path, not_exist_path


def compare_different_persons(person1_images, person1_name, person2_images, person2_name, failed_path):
    for image1 in person1_images:
        for image2 in person2_images:
            start_time = time.time()
            result,m,n = compare_images(image1, image2, ALGORITHM)
            debug_print("comparing time = " + str(time.time() - start_time))
            if result == SAME:
                debug_print("Oops. it said 2 different people are ###THE SAME###!")
                save_2_images_different_persons(failed_path, image1, image2, person1_name, person2_name)

            elif result == DIFFERENT:
                debug_print("Great! it said 2 different people are different!")


def compare_different_profiles(profile1, profile1_name, profile2_name, profile2, failed_path, not_exist_path,
                               resolution):
    start_time = time.time()
    resize.resize(profile2, resolution)

    # profile1_loc_faces = get_faces_loc(profile1, ALGORITHM)
    # profile1_faces = get_faces(profile1, ALGORITHM, profile1_loc_faces)
    # profile2_loc_faces = get_faces_loc(profile2, ALGORITHM)
    # profile2_faces = get_faces(profile2, ALGORITHM, profile2_loc_faces)

    profile1_faces = dnn.get_faces_from_image(profile1)
    profile2_faces = dnn.get_faces_from_image(profile2)

    # result,m,n = compare_images(profile1, profile2, ALGORITHM, face_locs1=profile1_loc_faces, face_locs2=profile2_loc_faces,
    #                         faces1=profile1_faces, faces2=profile2_faces)

    result,m,n = FN.compare_faces_FN(profile1_faces, profile2_faces)

    time_it_took = time.time() - start_time
    debug_print("comparing time = " + str(time.time() - start_time))

    if result == SAME:
        debug_print("Oops. it said 2 different people are ###THE SAME###!")
        save_2_images_different_persons(failed_path, profile1, profile2, profile1_name, profile2_name)
        save_faces_2_person(failed_path,profile1_faces[m], profile2_faces[n], profile1_name, profile2_name)

        return True, time_it_took

    elif result == DIFFERENT:
        debug_print("Great! it said 2 different people are different!")

    elif save:
        imsave(not_exist_path + profile2_name, profile2)

    return False, time_it_took


def compare_person_to_others_big(images, person_index, failed_path):
    person, person_images, person_name = get_person_from_images(images, person_index)

    num_of_people = len(images)

    for i in range(person_index + 1, num_of_people):
        curr_person, curr_person_images, curr_person_name = get_person_from_images(images, i)
        compare_different_persons(person_images, person_name, curr_person_images, curr_person_name, failed_path)
        if i % 5 == 0:
            print(i)


def compare_person_to_others_profile(images, person_index, failed_path, num_of_checks, not_exists_path, resolution):
    person, person_images, person_name = get_person_from_images(images, person_index)

    profile = get_profile(person_images)
    profile = resize.resize(profile, resolution)

    num_of_people = len(images)

    indexes_to_check = get_random_indexes(person_index, num_of_people, num_of_checks)

    true_negative = 0
    false_positive = 0

    checked = len(indexes_to_check)

    avg_compare_time = 0
    for i in indexes_to_check:
        curr_person, curr_person_images, curr_person_name = get_person_from_images(images, i)
        check_profile = get_profile(curr_person_images)
        compare_result, compare_time = compare_different_profiles(profile, person_name, curr_person_name,
                                                                  check_profile,
                                                                  failed_path,
                                                                  not_exists_path, resolution)
        avg_compare_time += compare_time
        if compare_result:
            false_positive += 1
        else:
            true_negative += 1

    avg_compare_time /= checked

    return true_negative, false_positive, avg_compare_time, checked


def small_tester(images, resolution):
    results_folder = "results_folder" + ALGORITHM + "/"
    compared_himself_folder = "compared_himself/"
    failed_folder = "failed_pictures/"
    not_exist_folder = "not_exist_faces/"
    compared_himself_path, failed_path, not_exist_path = create_results_dir(results_folder, compared_himself_folder,
                                                                            failed_folder, not_exist_folder)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    num_of_checks = 0
    num_of_peoples_took_profiles_from = 0

    # this keeps the data for running times in the comparing (it is needed to make a weighted avg)
    num_checks_avg_time_person = []
    num_checks_avg_time_person_others = []

    num_of_people = len(images)
    checked_with_others = 0
    for i in range(num_of_people):
        print(i)
        person, person_images, person_name = get_person_from_images(images, i)

        n = len(person_images)
        if n > 1:
            num_of_checks_in_person, result, avg_time_in_person = compare_person_to_himself_up_to_k(5, person_images,
                                                                                                    person_name,
                                                                                                    compared_himself_path,
                                                                                                    not_exist_path,
                                                                                                    resolution)
            num_checks_avg_time_person += [num_of_checks_in_person, avg_time_in_person]

            num_of_peoples_took_profiles_from += 1
            num_of_checks += num_of_checks_in_person
            if result:
                true_positive += 1
            else:
                false_negative += 1
            try:
                got_true_negative, got_false_positive, person_compare_time, checked = compare_person_to_others_profile(
                    images, i,
                    failed_path,
                    num_of_checks_in_person,
                    not_exist_path,
                    resolution)
                checked_with_others += checked
                num_checks_avg_time_person_others += [checked, person_compare_time]
                true_negative += got_true_negative
                false_positive += got_false_positive
            except:
                debug_print("")

    # calculate times:
    avg_compare_time_on_himself = get_avg_compare_time(num_checks_avg_time_person, num_of_checks)
    avg_compare_time_on_others = get_avg_compare_time(num_checks_avg_time_person_others, checked_with_others)

    return true_positive, true_negative, false_positive, false_negative, checked_with_others, num_of_peoples_took_profiles_from, avg_compare_time_on_others, avg_compare_time_on_himself


def resolution_tester(main_dir_path, resolutions):
    for resolution in resolutions:

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        check_others = 0
        check_himself = 0
        total_time_others = 0
        total_time_himself = 0
        avg_time_others = 0
        avg_time_himself = 0

        dirs = [dirs for subdir, dirs, files in os.walk(main_dir_path)]
        num_files = len(dirs)

        for start_index in range(0, num_files, num_images_load):
            print("start_index = ", start_index)
            images = load_images(main_dir_path, start_index)
            curr_TP, curr_TN, curr_FP, curr_FN, curr_CO, curr_CH, avg_time_others, avg_time_himself = small_tester(
                images, resolution)
            TP += curr_TP
            TN += curr_TN
            FP += curr_FP
            FN += curr_FN
            check_others += curr_CO
            check_himself += curr_CH
            total_time_others += avg_time_others * curr_CO
            total_time_himself += avg_time_himself * curr_CH
        try:
            # put it in precentages:
            FP /= check_others
            TN /= check_others
            FN /= check_himself
            TP /= check_himself
            avg_time_others = total_time_others / check_others
            avg_time_himself = total_time_himself / check_himself
        except:
            debug_print("no checks made")

        print_parameters(resolution, TP, TN, FP, FN, avg_time_others, avg_time_himself)


def big_tester(main_dir_path):
    images = load_images(main_dir_path)
    results_folder = "results_folder/"
    compared_himself_folder = "compared_himself/"
    failed_folder = "failed_pictures/"
    not_exist_folder = "not_exist_faces/"
    compared_himself_path, failed_path, not_exist_path = create_results_dir(results_folder, compared_himself_folder,
                                                                            failed_folder, not_exist_folder)

    num_of_people = len(images)
    for i in range(num_of_people):
        print(i)
        person, person_images, person_name = get_person_from_images(images, i)

        n = len(person_images)
        if n > 1:
            compare_person_to_himself(person_images, person_name, compared_himself_path, not_exist_path)

        compare_person_to_others_big(images, i, failed_path)


def delete_empty_dirs(main_dir_path):
    for start_index in range(0, len([dirs for subdir, dirs, files in os.walk(main_dir_path)]), num_images_load):
        print("start index = ", start_index)
        images = load_images(main_dir_path, start_index)
        for i in range(len(images)):
            person, person_images, person_name = get_person_from_images(images, i)
            if len(person_images) == 0:
                shutil.rmtree(main_dir_path + '/' + person_name)
                debug_print("delete = " + person_name)


random.seed(100)
# delete_empty_dirs(BIG_DS)
resolution_tester(TRY_DS, RESOLUTIONS)
