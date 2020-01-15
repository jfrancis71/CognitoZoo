(* ::Package:: *)

<<"Experimental/GenerativeModels/CZNBModels.m"


mnist=ResourceData["MNIST","TrainingData"][[;;500,1]];


binImages = ImageData/@Binarize/@mnist;


images = CZDiscretize/@mnist;


nbmodel1 = CZCreateNBModelBinary[];


nbtrain1 = CZTrain[ nbmodel1, binImages ];


CZLogDensity[ nbtrain1, binImages[[1]] ]


Image@CZSample[ nbtrain1 ]


nbmodel2 = CZCreateNBModelDiscrete[];


nbtrain2 = CZTrain[ nbmodel2, images ];


CZLogDensity[ nbtrain2, images[[1]] ]


Image@CZSample[ nbtrain2 ]


nbmodel3 = CZCreateNBModelRealGauss[];


nbtrain3 = CZTrain[ nbmodel3, ImageData/@mnist ];


CZLogDensity[ nbtrain3, ImageData@mnist[[1]] ]


Image@CZSample[ nbtrain3 ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


vaemodel1 = CZCreateVaEBinary[];


vaetrain1 = CZTrain[ vaemodel1, binImages ];


CZLogDensity[ vaetrain1, binImages[[1]] ]


Image@CZSample[ vaetrain1 ]


vaemodel2 = CZCreateVaEDiscrete[];


vaetrain2 = CZTrain[ vaemodel2, images ];


CZLogDensity[ vaetrain2, images[[1]] ]


Image@CZSample[ vaetrain2 ]


vaemodel3 = CZCreateVaERealGauss[];


vaetrain3 = CZTrain[ vaemodel3, ImageData/@mnist ];


Image@CZSample[ vaetrain3 ]


CZLogDensity[ vaetrain3, images[[1]] ]


<<"Experimental/GenerativeModels/CZPixelCNN.m"


cnnmodel1 = CZCreatePixelCNNBinary[];


cnntrain1 = CZTrain[ cnnmodel1, binImages ];


CZLogDensity[ cnntrain1, binImages[[1]] ]


Image@CZSample[ cnntrain1 ]


cnnmodel2 = CZCreatePixelCNNDiscrete[];


cnntrain2 = CZTrain[ cnnmodel2, images ];


CZLogDensity[ cnntrain2, images[[1]] ]


Image@CZSample[ cnntrain2 ]


<<"Experimental/GenerativeModels/CZPixelVaE.m"


cnnvaemodel1 = CZCreatePixelVaEBinary[];


cnnvaetrain1 = CZTrain[ cnnvaemodel1, binImages ];


CZLogDensity[ cnnvaetrain1, binImages[[1]] ]


Image@CZSample[ cnnvaetrain1 ]


cnnvaemodel2 = CZCreatePixelVaEDiscreteImage[];


cnnvaetrain2 = CZTrain[ cnnvaemodel2, images ];


CZLogDensity[ cnnvaetrain2, images[[1]] ]


Image@CZSample[ cnnvaetrain2 ]
