# Art-using-GANs
Final MTP(Major technical project) for BTech Computer at IIT Mandi,2020. 
Generating novel art using GAN (Generative adversarial networks) based on combination of genres and styles.

Please refer to the report for the details [Report](https://github.com/Kaustubh1Verma/Art-using-GANs/blob/master/MTP_Project.pdf)

## Objective and scope of the Work
Images are processed using a CNN, while the questions are processed using an LSTM.  These tensors are then decomposed into objects and fed as input into the RN module.
![Alt text](CLEVR.png?raw=true "Title")

The objective of our work is to do a thorough study and application of previous methods of generating novel art
and come up with a novel approach based on GAN that can either improve upon previous artworks or combine
their methodologies to better script the process of generating art. Two major pillars of our artwork are genre
and art,whereby we look to generate artwork for a particular genre and multiple styles. We aim to have a
final methodology that is capable enough to generating artworks that could be used across multiple genres and
styles with the availability of required datasets.\\
However it must be noted that the evaluation of our work is largely subjective rather than objective. Same
artwork can have different evaluation by different individuals. Also creative art can vary from being very
specific to very abstract, therefore we would like our generated artwork to fall somewhere in the middle of
both extremes.\\
Wikiart is the most prominent and heavily used artistic dataset available. In accordance
with the resources available to us and the dataset we could gather, we restrict our approach to generating
artwork for the genre ”Landscape”, and then transferring multiple styles to wrap up of artistic image.

## Resuls
Please refer to the report for detailed results.
Attaching some sample images generated by our architecture.
![alt text](https://github.com/Kaustubh1Verma/Art-using-GANs/blob/master/content_image_256.PNG)
![alt text](https://github.com/Kaustubh1Verma/Art-using-GANs/blob/master/style_transfered.PNG)
![alt text](https://github.com/Kaustubh1Verma/Art-using-GANs/blob/master/style_transfered2.PNG)



