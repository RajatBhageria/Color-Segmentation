import numpy as np
import os, cv2
import math
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage import morphology
from sklearn import linear_model
import pickle

maskFolder = 'Labeling/labeled_data/RedBarrel/'
imageFolder = 'Labeling/images/'

#train the model
def trainModel():
    # initialize the empty xtrains and ytrains for the color segmentation
    Xtrain = np.empty((0, 3))
    Ytrain = np.empty((0,))

    # figure out the total number of pixels inside the barrels and outside for the prior probability calc
    totalInPixels = 0
    totalOutPixels = 0

    # initialize for the distance regression on the barrels
    pixelHeightsXtrain = np.empty((0,1))
    pixelHeightsYtrain = np.empty((0,))

    # load all training data
    for filename in os.listdir(maskFolder):
        # Read one train mask and one image
        mask = np.load(os.path.join(maskFolder, filename))

        # get the name of the image
        indexNpy = filename.find(".npy")
        distanceString = filename[0:indexNpy]
        imageName = distanceString + '.png'
        image = cv2.imread(os.path.join(imageFolder, imageName))

        # convert the image to a HSV file
        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # get the region
        region = mask.astype(int)

        # find all the pixel indicies that are outside the mask and within the mask
        inPixels = (region == 1)
        outPixels = (region == 0)

        # get the HSV values for each of the in and out pixels
        inPixelHSV = img2[inPixels]
        outPixelHSV = img2[outPixels]

        # find the number of inpixels and outPixels
        numInPixels, _ = inPixelHSV.shape
        numOutPixels, _ = outPixelHSV.shape
        totalInPixels = totalInPixels + numInPixels
        totalOutPixels = totalOutPixels + numOutPixels

        # add the inpixels to the training set
        Xtrain = np.vstack((Xtrain, inPixelHSV))
        inPixelLabels = np.full((numInPixels,), 1)
        Ytrain = np.append(Ytrain, inPixelLabels)

        # add the oupixels to the training set
        Xtrain = np.vstack((Xtrain, outPixelHSV))
        outPixelLabels = np.full((numOutPixels,), 0)
        Ytrain = np.append(Ytrain, outPixelLabels)

        #add the distance metrics as well
        for regionProp in regionprops(region.astype(int)):
            # take regions with large enough areas
            if regionProp.area > 100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = regionProp.bbox
                pixelHeight = maxr-minr
                pixelHeightsXtrain = np.append(pixelHeightsXtrain,pixelHeight)
                pixelHeightsYtrain = np.append(pixelHeightsYtrain,float(distanceString))

    # convert Xtrains to nparrays
    Xtrain = np.asarray(Xtrain)
    Ytrain = np.asarray(Ytrain)

    numPixels, _ = Xtrain.shape

    # do cross validation using bagging
    numIterationsCV = 5

    # set up cross validation total matricies
    mu_barrel_all = np.zeros((numIterationsCV, 3))
    mu_others_all = np.zeros((numIterationsCV, 3))
    covariance_barrel_all = np.zeros((numIterationsCV, 3, 3))
    covariance_others_all = np.zeros((numIterationsCV, 3, 3))
    prior_barrel_all = np.zeros((numIterationsCV, 1))
    prior_others_all = np.zeros((numIterationsCV, 1))

    for i in range(0,numIterationsCV):
        #select which items we want to use -- keep 95% for testing
        numTesting = int(numPixels*.95)
        testItems = np.random.choice(numPixels,numTesting)
        XtrainCV = Xtrain[testItems]
        YtrainCV = Ytrain[testItems]

        #get the subset of the CV training set that's only the barrel vs not the barrel
        XtrainBarrel = XtrainCV[YtrainCV==1]
        XtrainOthers = XtrainCV[YtrainCV==0]

        #find all mus and sigmas and priors
        mu_barrel = np.mean(XtrainBarrel,axis=0)
        mu_others = np.mean(XtrainOthers,axis=0)
        covariance_barrel = np.cov(np.transpose(XtrainBarrel))
        covariance_others = np.cov(np.transpose(XtrainOthers))
        prior_barrel = (XtrainBarrel.shape[0]+0.0)/(XtrainBarrel.shape[0]+XtrainOthers.shape[0])
        prior_others = (XtrainOthers.shape[0]+0.0)/(XtrainBarrel.shape[0]+XtrainOthers.shape[0])

        # add to the all matrix to later take average
        mu_barrel_all[i] = mu_barrel
        mu_others_all[i] = mu_others
        covariance_barrel_all[i] = covariance_barrel
        covariance_others_all[i] = covariance_others
        prior_barrel_all[i] = prior_barrel
        prior_others_all[i] = prior_others

    #calculate the averages for the parameters across all the cross validations
    mu_barrel = np.mean(mu_barrel_all,axis=0)
    mu_others = np.mean(mu_others_all,axis=0)
    covariance_barrel = np.mean(covariance_barrel_all,axis=0)
    covariance_others = np.mean(covariance_others_all,axis=0)
    prior_barrel = np.mean(prior_barrel_all)
    prior_others = np.mean(prior_others_all)

    #save everything
    np.save("mu_barrel.npy", mu_barrel)
    np.save("mu_others.npy", mu_others)
    np.save("covariance_barrel.npy",covariance_barrel)
    np.save("covariance_others.npy",covariance_others)
    np.save("prior_barrel.npy",prior_barrel)
    np.save("prior_others.npy",prior_others)

    #do linear regression for the distance estimations
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(1.0/pixelHeightsXtrain.reshape(-1,1), pixelHeightsYtrain)

    #save the regression
    pickle.dump(regr, open('distanceRegression.sav', 'wb'))

    return mu_barrel, mu_others, covariance_barrel, covariance_others, prior_barrel, prior_others

#@param: image xi
#@param: barrel mask with 1 if pixel is barrel and 0 otherwise for each pixel in xi
def createBarrelMask(xi):
    #load the model with the mus, sigmas, and priors
    #if you want to re-run the model
    #mu_barrel, mu_others, covariance_barrel, covariance_others, prior_barrel, prior_others = trainModel()

    #if you want to load the data from elsewhere
    mu_barrel = np.load("mu_barrel.npy")
    mu_others = np.load("mu_others.npy")
    covariance_barrel = np.load("covariance_barrel.npy")
    covariance_others = np.load("covariance_others.npy")
    prior_barrel = np.load("prior_barrel.npy")
    prior_others = np.load("prior_others.npy")

    # Find the P(X|C) or the posteriors for multivariate gaussian
    exponent_barrel = np.einsum("rcx, xy, rcy -> rc",(xi-mu_barrel),covariance_barrel,(xi-mu_barrel))
    _,logDetBarrel = np.linalg.slogdet(2 * math.pi * covariance_barrel)
    liklihood_barrel = -.5 * logDetBarrel - .5 * exponent_barrel

    exponent_others = np.einsum("rcx, xy, rcy -> rc",(xi-mu_others),covariance_others,(xi-mu_others))
    _,logDetOthers = np.linalg.slogdet(2 * math.pi * covariance_others)
    liklihood_others = -.5 * logDetOthers - .5 * exponent_others

    #convert the liklihoods to predictions
    predictionBarrel = -1*liklihood_barrel + math.log(prior_barrel) + 9000000
    predictionOthers = -1*liklihood_others + math.log(prior_others) - 10000000

    #take the argmax to return the correct class
    barrelMask = np.zeros((xi.shape[0],xi.shape[1]))
    barrelMask[predictionBarrel>=predictionOthers] = 1

    return barrelMask

#actually classify each of the pixels in the image as a barrel or not a barrel
def barrelClassification(img):
    #convert the image to a HSV image
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #convert the image to a barrelMask
    barrelMask = createBarrelMask(img2).astype(bool)

    #apply masks to remove noise
    barrelMask = morphology.remove_small_holes(barrelMask,min_size=1000)
    barrelMask = ~barrelMask
    barrelMask = ndimage.morphology.binary_erosion(barrelMask,structure=np.ones((5,5)))
    barrelMask = ndimage.morphology.binary_opening(barrelMask,structure=np.ones((5,5)))

    #create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    #find the region where the barrel is, display bbox, find centroid, and find distance.
    x=0
    y=0
    d=0
    for region in regionprops(barrelMask.astype(int)):
        # take regions with large enough areas
        if region.area >100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            #draw the centroid
            y = np.average([minr,maxr])
            x = np.average([minc,maxc])
            plt.plot(x,y, 'ro')

            # find the distance of the barrel from the camera using linear regression.
            loaded_model = pickle.load(open('distanceRegression.sav', 'rb'))
            #find the pixel height of the barrel in the test image
            pixelHeight = maxr-minr
            #estimate the distance away from the camera
            d = float(loaded_model.predict(1.0/pixelHeight))

    #display the image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()

    return x,y,d

if __name__ == "__main__":
    image = cv2.imread(os.path.join(imageFolder, "2.2.png"))
    barrelClassification(image)
    #trainModel() #run this if you just to train the model and save the parameters as .npy files