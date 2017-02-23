
import cv2
# used to trained data to disk
import pickle
import cv2.cv as cv
import numpy
# perform various file operations like creating and listing directories
import os
# take command line arguments and proper termination of program
import sys
# to show a particular task execution time
import time

# http://stackoverflow.com/a/189664/4723940
class GetOutOfLoop(Exception) :
    pass

def majority(mylist) :
    '''
      takes a list and returns an element which has highest frequency in the given list.
    '''
    myset = set(mylist)
    ans = mylist[0]
    ans_f = mylist.count(ans)
    for i in myset :
        if mylist.count(i) > ans_f :
            ans = i
            ans_f = mylist.count(i)
    return ans

def get_positive_lst(mylist) :
    '''
      takes a list and returns the positive integers in the list
    '''
    # http://stackoverflow.com/a/23096436/4723940
    x = [num for num in mylist if num >= 0]
    return x

def save_object(obj, filename) :
    # http://stackoverflow.com/a/4529901/4723940
    with open(filename, 'wb') as output :
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def get_object(filename) :
    with open(filename, 'wb') as input :
        obj = pickle.load(input)
    return obj

def take_images_for_training(input_images, entity_name) :
    '''
      takes parameters:
        input_images-- the path to save training images to
        entity_name -- the name of the person
    '''
    print("Starting to take the training entity images ...")
    sttime = time.clock()
    i = 0
    path_to_save_images = input_images + os.path.sep
    capture = cv2.VideoCapture(0)
    print("keep your face in different angles")
    while(True) :
        ret , frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame',gray)
        print("writing to file " + str(i) + ".png")
        cv2.imwrite(path_to_save_images + str(entity_name + '_' + str(i)) + '.png', gray)
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
        i = i + 1
        if i >= 10 :
            break
        print("sleeping for five seconds")
        time.sleep(5)
    capture.release()
    cv2.destroyAllWindows()
    print("Successfully completed the task in %.2f Secs." % (time.clock() - sttime))

def detect_faces_in_image(input_images, output_faces) :
    '''
        takes two parameters: input_images and output_images
    '''
    print("Starting to detect faces in images and save the cropped images to output file...")
    sttime = time.clock()
    image_files = os.listdir(input_images)
    i = 0
    for filename in image_files :
        image_path = input_images + os.path.sep + filename
        print(image_path)
        color_img = cv2.imread(image_path)
        # print(color_img)
        # print(color_img.shape)
        if(color_img is None) :
            continue
        # converting color image to grayscale image
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		# find the bounding boxes around detected faces in images
        bBoxes = frontal_face.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
		# print(bBoxes)
        for box in bBoxes :
			# print(box)
			# crop and save the image at specified location
			# cropImage(color_img, box)
            [p, q, r, s] = box
        	# crop and save the image provided with the co-ordinates of bounding box
            write_img_color = color_img[q:q + s, p:p + r]
        	# saveCropped(write_img_color, name)
            cv2.imwrite(output_faces + os.path.sep + filename, write_img_color)
            i = i + 1
    print("Successfully completed the task in %.2f Secs." % (time.clock() - sttime))

def process_cropped_images(input_images, output_faces, subject_directory) :
    print("Starting to rearrange images in appropriate directories...")
    sttime = time.clock()
    image_files = os.listdir(output_faces)
    for filename in image_files :
        image_path = output_faces + os.path.sep + filename
        # assuming the invidual name will not have underscore character
        subject_name = filename.split('_')[0]
        new_path = subject_directory + os.path.sep + subject_name
        value = os.path.isdir(new_path)
        if not value :
            os.makedirs(new_path)
        new_path += os.path.sep + filename
        # moving file TODO: use Python operation instead of the hard coded value
        os.system("mv " + image_path + " " + new_path)
    print("Successfully completed the task in %.2f Secs." % (time.clock() - sttime))
    os.system("rm -rf " + input_images + " " + output_faces)

def get_images(path, size) :
    '''
    path: path to a folder which contains subfolders of for each subject/person
        which in turn cotains pictures of subjects/persons.

    size: a tuple to resize images.
        Ex- (256, 256)
    '''
    sub = 0
    images, labels = [], []
    people = []
    for subdir in os.listdir(path) :
        for image in os.listdir(path + os.path.sep + subdir) :
            img= cv2.imread(path + os.path.sep + subdir + os.path.sep + image, cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, size)
            images.append(numpy.asarray(img, dtype = numpy.uint8))
            labels.append(sub)
        people.append(subdir)
        sub += 1
    return [images, labels, people]

def train_model(path) :
    '''
    Takes path to images and train a face recognition model
    Returns trained model and people
    '''
    [images, labels, people] = get_images(path, (256, 256))
    labels= numpy.asarray(labels, dtype= numpy.int32)
    # initializing eigen_model and training
    print("Initializing LBPH FaceRecognizer and training...")
    sttime = time.clock()
    eigen_model = cv2.createLBPHFaceRecognizer()
    eigen_model.train(images, labels)
    print("Successfully completed training in " + str(time.clock() - sttime) + " seconds!")
    return [eigen_model, people]

def detect_faces(frontal_face, image) :
    '''
    Takes an image as input and returns an array of bounding box(es).
    '''
    bBoxes = frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    return bBoxes

def the_real_test(eigen_model, people, subject_directory, frontal_face) :
    '''
      INPUT: subject_directory named according to convention
             the Eigen Face Detection dataset
      OUTPUT: name of the person
    '''
    try :
        # starts recording video from camera and detects & predict subjects
        sttime = time.clock()
        cap = cv2.VideoCapture(0)
        counter = 0
        last_20 = [-1 for i in range(20)]
        final_5 = []
        box_text = "Subject: "
        while(True) :
            print(counter)
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.equalizeHist(gray_frame)
            bBoxes = detect_faces(frontal_face, gray_frame)
            for bBox in bBoxes :
                (p,q,r,s) = bBox
                cv2.rectangle(frame, (p, q), (p + r, q + s), (225, 0, 25), 2)
                crop_gray_frame = gray_frame[q:q+s, p:p+r]
                crop_gray_frame = cv2.resize(crop_gray_frame, (256, 256))
                [predicted_label, predicted_conf] = eigen_model.predict(numpy.asarray(crop_gray_frame))
                # print(predicted_conf)
                # if (predicted_conf / 100.0) > 40 and len(get_positive_lst(final_5)) == 5 :
                last_20.append(predicted_label)
                last_20 = last_20[1:] # queue
                cv2.putText(frame, box_text, (p-20, q-5), cv2.FONT_HERSHEY_PLAIN, 1.3, (25,0,225), 2)
                '''
                counter modulo x: changes value of final label for every x frames
                Use max_label or predicted_label as you wish to see in the output video.
                '''
                if (counter % 10) == 0 :
                    max_label = majority(get_positive_lst(last_20))
                    box_text = format("Subject: " + people[max_label])
                    # timeout after a particular interval
                    if counter > 20 :
                        final_5.append(max_label)
                        # it always takes max_label into consideration
                        print("[" + str(len(final_5)) + "] Detected face is: " + people[max_label])
                        if (predicted_conf / 100.0) > 80 and len(final_5) == 5 :
                        # if len(final_5) == 5 :
                            print("connection timed out...")
                            return True, people[max_label]
                            # raise GetOutOfLoop
                        else :
                            print("No matching faces recognized!")
                            return False, people[max_label]
                            # raise GetOutOfLoop
            # else :
            #     # print("no face detected..")
            #     last_20.append(-1)
            #     last_20 = last_20[1:] # queue
            #     # raise GetOutOfLoop
            cv2.imshow("Video Window", frame)
            counter += 1
            if (cv2.waitKey(5) & 0xFF == 27):
                break
            # print(last_20)
        endtime = (time.clock() - sttime)
        cap.release()
        cv2.destroyAllWindows()
    except GetOutOfLoop :
        pass

def create_reqd_dirs(input_images, output_faces, subject_directory) :
    value = os.path.isdir(input_images)
    if not value :
        os.makedirs(input_images)
    value = os.path.isdir(output_faces)
    if not value :
        os.makedirs(output_faces)
    value = os.path.isdir(subject_directory)
    if not value :
        os.makedirs(subject_directory)

def usage_instructions() :
    print("python main.py " + "<name of individual> " + "<test/train>")
    sys.exit(1)

if __name__ == "__main__" :
    if len(sys.argv) != 3 :
        usage_instructions()
    # paths to input and output images
    input_images = str(os.getcwd()) + os.path.sep + "input_images"
    output_faces = str(os.getcwd()) + os.path.sep + "output_images"
    subject_directory = str(os.getcwd()) + os.path.sep + "sub_images"
    # load pre-trained frontalface cascade classifier
    frontal_face = cv2.CascadeClassifier(str(os.getcwd() + os.path.sep + "haarcascade_frontalface_default.xml"))
    # required arguments
    entity_name = sys.argv[1]
    type_of_method = sys.argv[2]
    # simple condition check
    if type_of_method == "train" :
        # create required directories
        create_reqd_dirs(input_images, output_faces, subject_directory)
        # take images using camera
        take_images_for_training(input_images, entity_name)
        # crop the faces in images
        detect_faces_in_image(input_images, output_faces)
        # rename images to appropriate directories
        process_cropped_images(input_images, output_faces, subject_directory)
        # train the saved dataset
        # eigen_model, people = train_model(subject_directory)
        # save_object(eigen_model, subject_directory + os.path.sep + "eigen_model.tdata")
        # save_object(people, subject_directory + os.path.sep + "people.tdata")
    elif type_of_method == "test" :
        # eigen_model = get_object(subject_directory + os.path.sep + "eigen_model.tdata")
        # people = get_object(subject_directory + os.path.sep + "people.tdata")
        # train the saved dataset
        eigen_model, people = train_model(subject_directory)
        # print(eigen_model)
        print(people)
        status_of_state, p_name = the_real_test(eigen_model, people, subject_directory, frontal_face)
        if p_name == entity_name :
            print("Authorized")
        else :
            print("Not Authorized")
    else :
        usage_instructions()
