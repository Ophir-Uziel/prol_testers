from sklearn.cluster import DBSCAN


def cluster(embedds_list, threshold_to_be_the_same = 1.1):
    """
    :param embedds_list: the list of the vectors of the faces
    :param threshold_to_be_the_same: the max length between vectors in which they will be at the same group
    :return: a dict where every key represents a different person, and the key "lonely_face" represents people that
    were found alone
    """

    db = DBSCAN(eps=threshold_to_be_the_same, min_samples=0).fit(embedds_list)
    labels = db.labels_

    label_vectors = {}
    labels['lonely_faces'] = []
    for i in range(embedds_list):
        if labels[i] == -1:
            # i classified it as nise. it means it's a face that appear only once
            labels['lonely_faces'].append(embedds_list[i])
            continue

        if labels[i] not in label_vectors:
            label_vectors[i] = [embedds_list[i]]
        else:
            label_vectors[i].append(embedds_list[i])

    return label_vectors


def avg(person_vecs):
    """
    make an average to a vector (simple as that)
    :param person_vecs:
    :return:
    """
    vecs_num = len(person_vecs)
    if vecs_num == 0:
        print("NO FACES GOT")
        return
    vec_length = len(person_vecs[0])
    return_vec = [0 for i in range(vecs_num)]
    for vec in person_vecs:
        for i in range(vec_length):
            return_vec[i] += vec[i]

    for i in range(vec_length):
        return_vec[i] /= vecs_num

    return return_vec


def vecs_avg_list(people_list):
    """
    :param people_list: a list of lists containing vectors of the same person, sorted by the number of appearances he
    had.
    :return: a list in the same order it got it of the average of the vectors of each person
    """
    vec_avg_return_list = []
    for person_images in people_list:
        vec_avg_return_list.append(avg(person_images))
    return vec_avg_return_list


def list_persons(label_vectors_dict):
    """
    :param label_vectors_dict: a dict contains what "cluster" returns
    :return: a list of the avg of each group of preson's faces vectors, sorted by the number of faces of that person
    that were found
    """
    people_list = []
    for label in label_vectors_dict:
        if label == 'lonely_faces':
            for vec in label_vectors_dict[label]:
                people_list.append([vec])

        else:
            person_vectors = []
            for vec in label_vectors_dict[label]:
                person_vectors.append(vec)
            people_list.append(person_vectors)
    people_list.sort(key=len)
    return vecs_avg_list(people_list)
