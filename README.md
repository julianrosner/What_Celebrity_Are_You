# What Celebrity Are You?
### by Julian Rosner

What_Celeberity_Are_You brings you the fun of discovering what celebrity you, your friends, or your favorite fictional characters most resemble.

<div align="center">
  <a href="https://homes.cs.washington.edu/~rosner98/explanation_and_demo.mp4" target="_blank">
    <img width="540" height="298" src="https://homes.cs.washington.edu/~rosner98/thumbnail.jpg" alt="Link to Video Explanation">
  </a>
  <p>Video Explanation</p>
</div>

## Problem
This project began life as facial recognition software trained on a database of celebrities. Then one day while I was waiting for a network to train I thought "If I ask this network to predict which celebrity from the database I am, won't it return the celebrity I look the most like?" After it finished training, I did just that and was very pleased with the results. I started asking the learner to predict more and more of my friends and family until I realized this little distraction was too entertaining to pass up, and the project was changed to center around determining what celebrity a person most resembles. All the user has to provide to get a celebrity match is a photo centered around their face no smaller than 64x64 pixels.

Not to make too fine a point of it, but this tool does not match an input face to a single picture of a celebrity. Rather the model, hopefully, has an abstract sense of each celebrity’s facial features, which it uses to determine which celebrity is most likely to be the one pictured in the input. Long story short, it matches each input picture to an identity, not to another picture.

## Dataset
My model was trained on the Chinese University of Hong Kong's CelebA celebrity face picture database. Specifically, the aligned and cropped portion of it. This database contains over 200,000 pictures of over 10,000 celebrities. I initially chose this database because it felt representative of the sort of labeled data you might have for real people in a practical setting due to its roughness. For instance, the pictures of many of these identities span the entire person's life. This would ideally mean that a learner trained on the data would have some robustness to temporary features of a human face such as beards, hair style, wrinkles, sunglasses, etcetera. For the most part this seemed to hold true in practice. The data also contains many instances of faces from the side, from below, or at other unusual angles to add some robustness to orientation. 

Unfortunately, this dataset also had its drawbacks. Firstly, in exploring the data I found a couple instances of mislabeled faces, creating unwanted noise. Second, the individual pictures were labeled with id's corresponding to the pictured celebrity, but nowhere were the id's associated with names. This is why my project’s predictions output with photos unaccompanied by names because to get the names I would need to hand label over 10,000 (somewhat noisy) identities. 

## Models & Techniques
The underlying models for this project are three convolutional neural networks, each trained by me on a different subset of the CelebA database. Additionally, the convolutional neural networks' architecture was designed by me. Each network is trained and tested for facial recognition, meaning that you can present any of these conv. nn's a face and they will predict which celebrity from their training set that face belongs to. In practice, however, my models are meant to be given pictures of people they have never seen before and determine what celebrities these people most strongly resemble. You could argue that this is a form of transfer learning where the models are trained to determine what celebrities celebrities most resemble so they can later determine what celebrities ordinary people most resemble. 

But why did I make three of them? I realized early in the process that there is an inherent tension between the expressiveness of a celebrity matcher and the quality of its matches. The more celebrities it has seen, the more likely it is to have seen a celebrity who looks like you, but also the less likely it is to find the best match for any given person. To explore this tradeoff, I constructed three versions of the dataset: small, medium, and large. Small and medium feature 100 and 1,000 identities respectively, while large retains all of the original 10,177. I trained a conv. nn on each of the three sizes. I will explore this more in the results section, but I ended up most happy with the network trained on the medium dataset because it struck the perfect balance.

## Pre-Existing Tools
Pytorch was the main external tool I used in this project. It was used for building the conv. neural nets, loading data into them, training them, testing them, etc. Matplotlib was used for displaying and saving images of loss curves and celebrity face matchings. I believe numpy and panda also saw some use, either directly by me or as dependencies to the aforementioned.

## Implemented for Project
After downloading the dataset I discovered that all the images came mixed together in one massive directory having the id’s labeled separately in a giant text file. However, Pytorch’s ImageFolder demands each class be in a separate directory, so I needed to sort these 200,000+ images through some means. Additionally, I wanted to make small, medium, and large versions of the dataset. And lastly, I would need to partition each of them into training and test sets. To accomplish these three tasks I wrote three java programs. This data-wrangling code is not included in the project because its use would no longer be necessary for anyone wanting to use the "What Celebrity Are You" tool.

Actually included, however, are four pieces of code I've written, some with help from the lecture tutorial series. CelebrityConvNet.py defines the architecture and forward propagation algorithm of my convolutional neural networks. TrainUtil.py, believe it or not, is a utility to help with the training of the networks including loading data, training, and assessing accuracy. TrainNetwork.py trains a CelebrityConvNet with hard-coded annealing, computes the accuracy of the model, saves a graph of the loss over time, and saves the final network for future use. Lastly, the only piece of code a user would explicitly call is PredictFace.py. This program loads one of the three neural networks (small, medium, or large) saved by TrainNetwork.py and uses it to find the best celebrity match for each picture in the What_Celebrity_Are_You/data/predict/to_predict folder. For each picture in that folder it then displays and saves a graphic of the provided picture next to three example photos of the respective best celebrity match.

## Results
I think that it is reasonable to assume that the testing accuracy of each network on known celebrities is a good measure of how well that network finds matches for non-celebrities. The conv. nn trained on the small dataset tends to achieve around 82% test accuracy on images of any of its 100 celebrity identities. The medium dataset conv. nn gets around 51% test accuracy. The large dataset conv. nn gets around 1%. The dramatic drop off between medium and large can be explained by the fact that I noticed early on that different network architectures optimized test accuracy for different datasets. Because the medium dataset conv. nn was already very expressive with its 1,000 known identities and the large dataset conv. nn took more than 24 hours to train on my laptop, I chose to optimize my architecture for the medium dataset, so that's where all the time, attention, and fine tuning went. You may have noticed that the small dataset conv. nn was even more test accurate than its medium equivalent. This is true, but only knowing 100 celebrities makes its predictions very inexpressive meaning there are many faces for which it can't find a good match simply because no similar-looking celebrity exists in its dataset. This is why I'm happiest with the medium conv. nn.

A picture is worth a thousand words when it comes to appreciating the results of this project. Let's take a look at some celebrity face matches found by my medium dataset trained model:

<div align="center">
<img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//medium_natalie.jpg?raw=true">
</div>

<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//copy.jpg?raw=true">
</div>

<div align="center">
  <img src="https://github.com/julianrosner/What_Celebrity_Are_You/blob/main/figs/medium_mom.jpg?raw=true">
</div>

Now let's look at some particular examples that demonstrate noteworthy features of the model:

<div align="center">
  <img src="https://github.com/julianrosner/What_Celebrity_Are_You/blob/main/figs/medium_paul1.jpg?raw=true">
</div>
In the above image, the man on the left is my father. Note that he and I both match to the same celebrity. The model has noticed the resemblance between us.

<div align="center">
  <img src="https://github.com/julianrosner/What_Celebrity_Are_You/blob/main/figs/medium_paul2.jpg?raw=true">
</div>
The above match and the one preceding it together demonstrate resistance to temporary appearance-altering features, a beard in this case.

<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//medium_lisa.jpg?raw=true">
</div>
In the above, my model has found a convincing celebrity match from a painting of a woman. This shows my network doesn't just work well on photographs.

<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//small_anakin.jpg?raw=true">
</div>
This is my favorite example. In the above, I asked my network to find a match for a stylized cgi model of the Star Wars character Anakin Skywalker. The celebrity match it found was Jake Lloyd, the actor who portrayed Anakin in the character's debut appearance. I find it remarkable that my network discovered a relationship between the two.

## Discussion
The greatest challenges of this project both came from the limitations of the database. All that data-wrangling code I mentioned before was surprisingly time consuming and not very engaging to write. Even worse, before I had wrangled any data, when all 200,000+ celebrity images were mixed together in a giant folder I went through the trouble of figuring out the names of 100 celebrities and their associated id’s in the dataset. This took a lot longer than you would probably imagine because many of the 10,000 celebrities in the database really aren’t that famous and reverse image search would struggle to identify them. I did all of this because I wanted to be able to output the celebrity’s name with each match, but the dataset did not provide them. Ultimately this was all wasted time because the 100 celebrity trained network was very inexpressive as I mentioned before, so I didn’t bother implementing the name feature for it  and there was no way I was going to repeat that whole process again for the 1,000 celebrity database.

Other programs out there address the same task as this project, but many of their celebrity matches are shallow. They often just compute the pixel difference between the input and a series of photos, one for each celebrity. They then output the celebrity photo with the least pixel difference. This is a very naive approach that is overly sensitive to lighting conditions, facial expression, orientation, and temporary features like beards or glasses. Such programs rarely identify an inputted celebrity as looking most like themself. My model is much more robust to the aforementioned factors and much more accurate in general, so I'd say my project benefits from its sophistication. 

Overall, I am very happy with my small and medium dataset trained convolutional neural nets. From a mathematical perspective, most human faces are quite similar to each other, so the fact that my medium dataset model was able to achieve a testing accuracy around 50% on a set of 1,000 different identities feels like a real achievement. Additionally, I am pleased with the models' ordinary face to celebrity matching ability as shown in the images above, and the models’ robustness to changes in the hairstyles, beards, ages, etc. of its subjects. The only adjustment I would make if I had more time would be to look into the issue of the large dataset trained conv. nn's inaccuracy, but with training it for even a couple of epochs taking several hours, any feedback to adjustments comes painfully slowly. The medium dataset trained conv. nn still delivers great results though, so I consider the project a complete success.

## Thanks for Reading
No privacy was violated in the making of this ReadMe. Each of the non-celebrities pictured above is a friend or family member who consented to the use of their picture in the document or video with the exception of Professor Redmon who I hope will forgive me. 
