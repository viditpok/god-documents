import numpy as np
import matplotlib.image as mpl_img
from PIL import Image
import os


class EigenFaces:
    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Load the images and store the data as required.
        """
        self.images_root_directory = images_root_directory
        self.subject_1_images = [
            os.path.join(images_root_directory, f"subject01.{name}.gif")
            for name in [
                "glasses",
                "happy",
                "leftlight",
                "noglasses",
                "normal",
                "rightlight",
                "sad",
                "sleepy",
                "surprised",
                "wink",
            ]
        ]
        self.subject_2_images = [
            os.path.join(images_root_directory, f"subject02.{name}.gif")
            for name in [
                "glasses",
                "happy",
                "leftlight",
                "noglasses",
                "normal",
                "rightlight",
                "sad",
                "sleepy",
                "surprised",
                "wink",
            ]
        ]
        self.test_images = {
            "subject01": os.path.join(images_root_directory, "subject01-test.gif"),
            "subject02": os.path.join(images_root_directory, "subject02-test.gif"),
        }

    def process_images(self, image_paths):
        data_matrix = None
        for path in image_paths:
            try:
                img = np.array(Image.open(path))
                resized_image = img[::4, ::4]
                vectorized_image = resized_image.flatten().reshape(1, -1)
                if data_matrix is None:
                    data_matrix = vectorized_image
                else:
                    data_matrix = np.vstack((data_matrix, vectorized_image))
            except FileNotFoundError:
                print("File {path} not found")

        mean_vector = np.mean(data_matrix, axis=0) if data_matrix is not None else None
        return data_matrix, mean_vector

    def calulate_eigenfaces(self, data_matrix, mean_vector):
        standardized_matrix = (data_matrix - mean_vector).T
        U, _, _ = np.linalg.svd(standardized_matrix, full_matrices=False)
        return U

    def calculate_proj_residuals(self, test_image, mean_vector, eigenfaces):
        standardized_test = test_image - mean_vector
        residuals = []
        for i in range(eigenfaces.shape[1]):
            projection = np.dot(standardized_test, eigenfaces[:, i]) * eigenfaces[:, i]
            residual = np.linalg.norm(standardized_test - projection) ** 2
            residuals.append(residual)
        return residuals

    def run(self):
        """
        This method calculates the eigen faces for both the subjects.
        It also calculates the projection residual for both test images.

        Output:
            map which consists of the following attributes
                1. "subject_1_eigen_faces"
                    numpy array of shape (6, a, b)
                    A `plt.imshow(map['subject_1_eigen_faces'][0])` should display first in a eigen face for subject 1.
                2. subject_2_eigen_faces
                    numpy array of shape (6, a, b)
                    A `plt.imshow(map['subject_2_eigen_faces'][0])` should display first in a eigen face for subject 2.
                3. s{ij}, projection residual for test image j and eigen faces for subject i.
        """

        S1, S1_mean = self.process_images(self.subject_1_images)
        S2, S2_mean = self.process_images(self.subject_2_images)

        U1 = self.calulate_eigenfaces(S1, S1_mean)
        U2 = self.calulate_eigenfaces(S2, S2_mean)

        subject_1_eigen_faces = np.array([U1[:, i].reshape(61, 80) for i in range(6)])
        subject_2_eigen_faces = np.array([U2[:, i].reshape(61, 80) for i in range(6)])

        subject_1_eigen_faces_flat = U1[:, :6]
        subject_2_eigen_faces_flat = U2[:, :6]

        test_image1 = np.array(Image.open(self.test_images["subject01"]))[
            ::4, ::4
        ].flatten()
        test_image2 = np.array(Image.open(self.test_images["subject02"]))[
            ::4, ::4
        ].flatten()

        s11 = self.calculate_proj_residuals(
            test_image1, S1_mean, subject_1_eigen_faces_flat
        )
        s12 = self.calculate_proj_residuals(
            test_image2, S1_mean, subject_1_eigen_faces_flat
        )
        s21 = self.calculate_proj_residuals(
            test_image1, S2_mean, subject_2_eigen_faces_flat
        )
        s22 = self.calculate_proj_residuals(
            test_image2, S2_mean, subject_2_eigen_faces_flat
        )

        return {
            "subject_1_eigen_faces": subject_1_eigen_faces,
            "subject_2_eigen_faces": subject_2_eigen_faces,
            "s11": s11,
            "s12": s12,
            "s21": s21,
            "s22": s22,
        }


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from PIL import Image
import os


# part a
def part_a(eigen_faces_results):
    subject_1_eigen_faces = eigen_faces_results["subject_1_eigen_faces"]
    subject_2_eigen_faces = eigen_faces_results["subject_2_eigen_faces"]

    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.imshow(subject_1_eigen_faces[i], cmap="gray")
        plt.title(f"Subj 1 - Eig {i+1}")
        plt.axis("off")

    for i in range(6):
        plt.subplot(2, 6, i + 7)
        plt.imshow(subject_2_eigen_faces[i], cmap="gray")
        plt.title(f"Subj 2 - Eig {i+1}")
        plt.axis("off")

    plt.suptitle("Top 6 Eigenfaces for Each Subject")
    plt.show()


# part b
def part_b(eigen_faces_results):
    s11 = np.sum(eigen_faces_results["s11"])
    s12 = np.sum(eigen_faces_results["s12"])
    s21 = np.sum(eigen_faces_results["s21"])
    s22 = np.sum(eigen_faces_results["s22"])

    print(f"Projection Residuals:")
    print(f"Subject 1 Test Image 1 (s11): {s11}")
    print(f"Subject 1 Test Image 2 (s12): {s12}")
    print(f"Subject 2 Test Image 1 (s21): {s21}")
    print(f"Subject 2 Test Image 2 (s22): {s22}")

    if s11 < s21:
        print("Test Image 1 likely belongs to Subject 1")
    else:
        print("Test Image 1 likely belongs to Subject 2")

    if s22 < s12:
        print("Test Image 2 likely belongs to Subject 2")
    else:
        print("Test Image 2 likely belongs to Subject 1")


eigen_faces = EigenFaces()
results = eigen_faces.run()

print("Running Part A: Visualizing Eigenfaces")
part_a(results)

print("Running Part B: Classifying Test Images")
part_b(results)
