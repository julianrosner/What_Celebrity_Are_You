# What Celebrity Are You?
## by Julian Rosner

What_Celeberity_Are_You brings users the fun of discovering what celebrity they, their friends, or their favorite fictional characters most resemble.


<div align="center" class="cropped" style="width:150px;height:10px;overflow:hidden;border:5px solid black;">
  <img style="width: 200px;height: 200px;"
   src="https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//medium_me.jpg"
    >
</div>


This project began life as a piece of facial recognition software trained on a database of celebrities. The one day while I was waiting for the network to train I thought "If I ask it to predict which celebrity I am, wouldn't it return the celebrity I most resemble? That sounds fun." Very pleased with the results, I started asking the learner to predict on more and more of my friends and family until I realized this little distraction was too entertaining to pass up, and the course of the entire project was changed to accomodate. 

The underlying model of this project is a convolutional neural network with arcjitecture designed by me. Pytorch was the main tool used in this project, but numpy, matplotlib, and panda also saw use.

My model was trained on the Chinese University of Hong Kong's CelebA database. Specifically, the aligned and cropped portion of it. This database contains a total of over 200,000 pictures of over 10,000 celebrities. I initially chose this database because it felt representative of the sort of labeled data you might have for real people in a practical setting due to some of its quirkiness. For instance, the pictures of many of these identities span the entire person's life. This would ideally mean that a learner trained on the data would have some robustness to temporary features of a human face such as beards, hair style, wrinkles, etecetera. For the most part this seemed to hold true in practice. The data also contains many instances of faces from the side, from below, or other unusual angles to add some robustness to orientation. 

Unfortuantely, this dataset also had its drawbacks. Firstly, in exploring the data I found a couple insatnces of mislabeled faces, creating unwanted noise. Second, the individual picture were labeled with id's corresponding to the pictured celebrity, but nowhere were the id's associated to names. This is why my project labels prediction outputs with photos unaccompanied by names because to get the names I would need to hand label over 10,000 (somewhat noisy) identities. 

To explore the inherent tension between the number of classes and the accuracy of a convultional neural network classifier, I constructed three versions of the dataset: small, medium, and large. Small and medium feature 100 and 1000 identities respectively, while large retains all of the original 10,177. I will explore this more in the results section, but I ended up most happy with the network trained on medium because it is far more expressive than the one trained on small and much more accurate than the one trained on large.

The first code I wrote for this project was a java program to reorganize the files of the dataset to accomodate the pytorch's ImageFolder dataset model. The images came altogether in one giant directory whith the identities labeled in a giant text file, so it was necessary to move each identity into its own separate folder so pytorch's ImageFolder could see the isolated classes. Next I wrote another java program to move a random tenth of the data into a separate test set. Lastly I wrote a third java program to create the small and medium datasets as a subset of the large. This code is not included as its use would no longer be necessary for anyone wanting to used the "What Celebrity Are You" tool and also because they are frankly not very presentable.

Actually included however are four pieces of code I've written, some with help from the lecture tutorial series. CelebrityConvNet.py defines the architecture and forward propogation algorithm of my convolutional neural network. TrainUtil.py, believe it or not, is a utililty to help with the training of the network including loading data, training, and assessing accuracy. TrainNetwork.py trains a CelebrityCOnvNet with hard-coded annealing, computes the accuracy of the model, saves a graph of the loss over time, and saves the final network for future use. Lastly, the piece of code a user would explicitly call is PredictFace.py. This program loads one of the three neural networks (small, medium, or large) saved by TrainNetwork.py and uses it to find the best celebrity match for each face in What_Celebrity_Are_You/data/predict/to_predict. For each face it then displays and saves a graphic of the provided picture next to three example photos of the matching celebrity.

Results
#GRAPHS HERE?

I think anyone would agree that a tool which tells people what celebrities they look like, if programmed well, should match unseen pictures of known celebrities to themselves. That is to say, the testing accuracy of this tool on known celebrities is a good measure of how well this tool finds matches for non-celebrities. The testing accuracy for the nn trained on the small dataset tends to achieve around 82% test accuracy. The medium dataset nn gets around 51%. The large dataset nn gets around 1%. The dramatic drop off between medium and large can be explained by the fact that I noticed early on that different network achitectures optimized test accuracy for different datasets. Because the medium dataset nn was already very exprrssive with its 1000 known identities and the large dataset nn took more than 24 hours to train on my laptop, I chose to optimize my architecture for the medium dataset, so that's where all the time and attention and fine tuning went. You may have noticed that the small datadet nn was quite accuracte too, but only knowing 100 celebrities makes its predictions very unexpressive meaning there are many faces for which it can't find a good match. The medium dataset nn is in the perfect goldilocks zone, however. 

Let's take a look at some celebrity face matches found by my model and see what we can learn from them.

  test accuracy on actual celebs
![Anakin_Example](https://github.com//julianrosner//What_Celebrity_Are_You//blob//main//figs//small_anakin.jpg?raw=true)
