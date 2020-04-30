"""[Module for image extraction inside the card design"""
import cv2
import numpy as np
import base64
import requests
import os


class ImageExtraction:

    def FindPoints(self, coord1, coord2, for_image=None):
        """[Finds the intersecting bounding boxes]

        Arguments:
            coord1 {[tuple]} -- [Coordinates of rectange]
            coord2 {[tuple]} -- [Coordinates of rectange]
        Keyword Arguments:
            for_image {[Boolean]} -- [Is intersection is deterined for image] (default: {None})

        Returns:
            [Boolean] -- [True/False]
        """
        x5 = max(coord1[0], coord2[0])
        y5 = max(coord1[1], coord2[1])
        x6 = min(coord1[2], coord2[2])
        y6 = min(coord1[3], coord2[3])
        if (x5 > x6 or y5 > y6):
            return False

        if for_image:
            return True
        else:

            intersection_area = (x6 - x5) * (y6 - y5)
            point1_area = (coord1[2] - coord1[0]) * (coord1[3] - coord1[1])
            point2_area = (coord2[2] - coord2[0]) * (coord2[3] - coord2[1])
            if intersection_area / point1_area > 0.55 or intersection_area / point2_area > 0.55:

                return True
            else:
                return False

    def detect_image(self, image=None, detected_coords=None, pil_image=None):
        """[Returns the Detected image coordinates]

        Keyword Arguments:
            image {[open-cv]} -- [input opencv image] (default: {None})
            detected_coords {[list of tuple]} -- [list of detected object's coordinates from faster rcnn model] (default: {None})
            pil_image {[pil_image]} -- [PIL input image] (default: {None})

        Returns:
            [list] -- [image points]
        """
        image_points = []
        # pre processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(dst, (5, 5), 0)
        ret, im_th = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
        # Set the kernel and perform opening
        # k_size = 6
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
        # edge detection
        edged = cv2.Canny(opened, 0, 255)
        # countours
        _, contours, hierarchy = cv2.findContours(
            edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # get the coords of the contours
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            image_points.append((x, y, x + w, y + h))

        for i in range(len(image_points)):
            for j in range(len(image_points)):
                if j < len(image_points) and i < len(image_points):
                    box1 = [float(c) for c in image_points[i]]
                    box2 = [float(c) for c in image_points[j]]
                    intersection = self.FindPoints(box1, box2, for_image=True)
                    contain = (
                        float(
                            box2[0]) <= box1[0] +
                        5 <= float(
                            box2[2])) and (
                        float(
                            box2[1]) <= box1[1] +
                        5 <= float(
                            box2[3]))
                    if intersection or contain:
                        if box1 != box2:
                            if box1[2] - box1[0] > box2[2] - box2[0]:
                                del image_points[j]
                            else:
                                del image_points[i]

        # extarct the points that lies inside the detected objects coords [
        # rectangle ]
        included_points_positions = [0] * len(image_points)
        for point in image_points:
            for p in detected_coords:
                contain = (float(p[0]) <= point[0] + 5 <= float(p[2])
                           ) and (float(p[1]) <= point[1] + 5 <= float(p[3]))
                if contain:
                    included_points_positions[image_points.index(point)] = 1
        # now get the image points / coords that lies outside the detected
        # objects coords
        image_points1 = []
        for point in image_points:
            if included_points_positions[image_points.index(point)] != 1:
                image_points1.append(point)
        image_points = sorted(set(image_points1), key=image_points1.index)
        # =image_points[:-1]
        width, height = pil_image.size
        widths = [point[2] - point[0] for point in image_points]
        if widths:
            position = widths.index(max(widths))
            if max(widths) - width <= 10:
                del image_points[position]

        return image_points

    def image_crop_get_url(self, coords=None, image=None):
        """[Returns the Image urls]

        Arguments:
            coords {[list]} -- [image points] (default: {None})
            image {[pil image]} -- [input image] (default: {None})

        Returns:
            [list] -- [list of image urls]
            [list] -- [list of image sizes]
        """
        images_urls = []
        images_sizes = []
        for coords in coords:
            cropped = image.crop((coords[0], coords[1], coords[2], coords[3]))
            images_sizes.append(cropped.size)
            cropped.save("image_detected.png")
            img = open("image_detected.png", "rb").read()
            base64_string = base64.b64encode(img).decode()
            url = "https://post.imageshack.us/upload_api.php"
            payload = {'key': '0346ANQUe74917fd7160ababf178d69779a76c7c',
                       'format': 'json',
                       'tags': 'sample',
                       'public': 'yes'}
            files = [
                ('fileupload', open('image_detected.png', 'rb'))
            ]
            response = requests.request("POST", url, data=payload, files=files)
            images_urls.append(response.json().get(
                "links", {}).get("image_link", ''))
            # images_urls.append("")
        if os.path.exists('image_detected.png'):
            os.remove('image_detected.png')
        return images_urls, images_sizes
