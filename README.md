# Cuda_IP_Akshat
Converting a normal image to a polaroid style image using (cuda npp image processing)

Madeby- Saksham Wadhera

Desc: The code takes in 10 ppm format images named img1 to img10  and convert them into polaroids.As there is no direct conversion i have studied how normal images can be edited to look like polaroids.For this i have done three implementations on the image--->
1.Firstly i have converted image to greyscale (pgm format), to have a classy old camera look.
2.Then i have cropped the image to polaroid dimesion ,which is genrally a sqaure.I have done this such that it crops from the middle of the image.
3.Then i tampered with highlights and shadows of the images to attain a polaroid effect.Note you change this according to your liking .I have commented it (just find something named shadows ).

How to run?
XX:The main code is inside the file"ImageCropper" in folder boxFilterNPP:XX(using cuda at scale for the enterprise lab week 5 (boxfilter), run in that environment if any error)

1.type "cd boxFilterNPP"  to change the directory to code directory
2.write "make clean build" if auto run of python do not work for some reason.
3.Finally type "make run" to run the code 
4.You will find output images in the folder "processed_images" inside the boxFilterNpp folder
5.if you want to see the output images download a software to view ppm images or just convert from online using converters.


Note::::Make Sure you have all neccessary libraries installed! I have edited the previously present files in the sample labratory(week 5 cuda scale enterprise) hence some files may be useless and names might be misguiding .I have clearly mentioned the useful files in the read me .Please do not delete any files -cause who knows it may lead to failure of code .Thanks!""



update!!!:Removed files img7.ppm and img10.ppm due to size and modified the code to work for the rest ,for adding them download from google drive folder and modify the code to include the remaining ones----"Thanks!";
