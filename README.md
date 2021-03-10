# What Celebrity Are You?
### by Julian Rosner

What_Celeberity_Are_You brings users the fun of discovering what celebrity they, their friends, or their favorite fictional characters most resemble.


<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//medium_me.jpg?raw=true">
  <p>me and my match</p>
</div>

This project began life as facial recognition software trained on a database of celebrities. Then one day while I was waiting for a network to train I thought "If I ask it to predict which celebrity from the database I am, wouldn't it return the celebrity I most resemble? That sounds fun." After it finished training, I did just that and was very pleased with the results, I started asking the learner to predict on more and more of my friends and family until I realized this little distraction was too entertaining to pass up and the project was changed to accomodate this. 

## Techniques
The underlying model of this project is three convolutional neural network, each trained by me on a different subset of the CelebA database. Additionally, the convolutional neural networks' architecture was designed by me. Each network is trained and tested for facial recognition, meaning that you can present any of its conv. nn's a face and it will predict which person from its training set that face belongs to. In pracitce, however, my model is meant to be given pictures of people it has never seen before and determine what celebrities these people most strongly resemble. You could argue that this is a form of transfering learning where the model is trained to determine what celebrities a set of celebrities most resemble so it can then be used to determine what celebrities ordinary people most resemble.

## Tools
Pytorch was the main external tool I used in this project. It was used for building the conv. neural nets, loading data into them, training them, testing them, etc. Matplotlib was used for displaying and saving images of loss curves and celebrity face matchings. I believe numpy and panda also saw some use, either directly by me or as dependencies to the aforementioned.


## Data
My model was trained on the Chinese University of Hong Kong's CelebA database. Specifically, the aligned and cropped portion of it. This database contains a total of over 200,000 pictures of over 10,000 celebrities. I initially chose this database because it felt representative of the sort of labeled data you might have for real people in a practical setting due to some of its quirkiness. For instance, the pictures of many of these identities span the entire person's life. This would ideally mean that a learner trained on the data would have some robustness to temporary features of a human face such as beards, hair style, wrinkles, etecetera. For the most part this seemed to hold true in practice. The data also contains many instances of faces from the side, from below, or other unusual angles to add some robustness to orientation. 

Unfortuantely, this dataset also had its drawbacks. Firstly, in exploring the data I found a couple insatnces of mislabeled faces, creating unwanted noise. Second, the individual picture were labeled with id's corresponding to the pictured celebrity, but nowhere were the id's associated to names. This is why my project labels prediction outputs with photos unaccompanied by names because to get the names I would need to hand label over 10,000 (somewhat noisy) identities. 

To explore the inherent tension between the number of classes and the accuracy of a convultional neural network classifier, I constructed three versions of the dataset: small, medium, and large. Small and medium feature 100 and 1000 identities respectively, while large retains all of the original 10,177. I will explore this more in the results section, but I ended up most happy with the network trained on medium because it is far more expressive than the one trained on small and much more accurate than the one trained on large.

## Code
The first code I wrote for this project was a java program to reorganize the files of the dataset to accomodate the pytorch's ImageFolder dataset model. The images came altogether in one giant directory whith the identities labeled in a giant text file, so it was necessary to move each identity into its own separate folder so pytorch's ImageFolder could see the isolated classes. Next I wrote another java program to move a random tenth of the data into a separate test set. Lastly I wrote a third java program to create the small and medium datasets as a subset of the large. This code is not included as its use would no longer be necessary for anyone wanting to used the "What Celebrity Are You" tool and also because they are frankly not very presentable.

Actually included however are four pieces of code I've written, some with help from the lecture tutorial series. CelebrityConvNet.py defines the architecture and forward propogation algorithm of my convolutional neural network. TrainUtil.py, believe it or not, is a utililty to help with the training of the network including loading data, training, and assessing accuracy. TrainNetwork.py trains a CelebrityCOnvNet with hard-coded annealing, computes the accuracy of the model, saves a graph of the loss over time, and saves the final network for future use. Lastly, the piece of code a user would explicitly call is PredictFace.py. This program loads one of the three neural networks (small, medium, or large) saved by TrainNetwork.py and uses it to find the best celebrity match for each face in What_Celebrity_Are_You/data/predict/to_predict. For each face it then displays and saves a graphic of the provided picture next to three example photos of the matching celebrity.

## Results

I think anyone would agree that a tool which tells people what celebrities they look like, if programmed well, should match unseen pictures of known celebrities to themselves. That is to say, the testing accuracy of this tool on known celebrities is a good measure of how well this tool finds matches for non-celebrities. The testing accuracy for the nn trained on the small dataset tends to achieve around 82% test accuracy. The medium dataset nn gets around 51%. The large dataset nn gets around 1%. The dramatic drop off between medium and large can be explained by the fact that I noticed early on that different network achitectures optimized test accuracy for different datasets. Because the medium dataset nn was already very exprrssive with its 1000 known identities and the large dataset nn took more than 24 hours to train on my laptop, I chose to optimize my architecture for the medium dataset, so that's where all the time and attention and fine tuning went. You may have noticed that the small datadet nn was quite accuracte too, but only knowing 100 celebrities makes its predictions very unexpressive meaning there are many faces for which it can't find a good match. The medium dataset nn is in the perfect goldilocks zone, however. 

Let's take a look at some celebrity face matches found by my model:

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
The man on the left of the above image is my father. Note that he and I both match to the same celebrity. The model has noticed the resmebelance between us.

<div align="center">
  <img src="https://github.com/julianrosner/What_Celebrity_Are_You/blob/main/figs/medium_paul2.jpg?raw=true">
</div>
The above match and the one preceding demonstrate the resistance to temporary appearance-altering features that I mentioned earlier, a beard in this case.

<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//medium_lisa.jpg?raw=true">
</div>
In the above, my model has found a convincing celebrity match from a painting of a woman. This shows my network doesn't just work well on photographs.

<div align="center">
  <img src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//small_anakin.jpg?raw=true">
</div>
In the above, I asked my model to find a match for a stylized cgi model of the Star Wars character Anakin Skywalker. The celebrity match found by my model is Jake Lloyd, the actor who potrayed Anakin in the character's debut appearance. My model managing to find this connection is remarkable to me.

## Conclusions
Overall, I am very happy with my small and medium dataset convultional neural nets. From a mathematical perspective, most human faces are quite similar to each other, so the fact that my medium dataset model was able to achieve a testing accuracy around 50% on a set of 1000 different identities feels like a real achievment. Additionally, I am very pleased with the models' celebrity matching ability as shown in the images above, and their robustness to the hair style, beard, and age of its subjects. The only adjustment I might make if I had more time is to look into the issue of the large dataset nn's innaccuaracy, but with training it for even a couple of epochs taking several hours any feedback to adjustments comes painfully slowly. The medium dataset nn still delivers great results though, so I consider the project a complete success.


## Thanks for Reading
<div align="center">
  <img src="https://github.com/julianrosner/What_Celebrity_Are_You/blob/main/figs/small_joseph.jpg?raw=true">
</div>
