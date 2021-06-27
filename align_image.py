import dlib
from utils.align_all_parallel import align_face

#Align Image
def run_alignment(image_path):
    predictor = dlib.shape_predictor("./utils/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


image_path='./input/zy.png'
input_image = run_alignment(image_path)

input_image = input_image.resize((1024, 1024))

print(input_image)

input_image.save('./input/1.png')