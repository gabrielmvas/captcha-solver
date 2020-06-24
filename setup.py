import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
     name='captcha_solver_object_detection',  
     version='10.2',
     license='MIT',
     author="Gabriel Vasconcelos",
     author_email="gabrielmvas@gmail.com",
     description="A captcha solver package with object detection",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/gabrielmvas/captcha-solver-object-detection",
     download="https://github.com/gabrielmvas/captcha-solver-object-detection/archive/10.2.tar.gz",
     packages = ['captcha_solver_object_detection'], 
     package_data={'captcha_solver_object_detection': ['model/*.*']},
     install_requires=[
          'tensorflow-object-detection-api',
          'numpy',
          'opencv-python',
          'pandas',
          'tensorflow'
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )