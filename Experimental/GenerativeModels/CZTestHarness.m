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


cnnmodel2 = CZCreatePixelCNNBinaryImage[];


cnntrain2 = CZTrain[ cnnmodel2, binImages ];


Image@CZSample[ cnntrain2 ]


CZLogDensity[ cnntrain2, binImages[[1]] ]


cnnmodel3 = CZCreatePixelCNNDiscreteImage[];


cnntrain3 = CZTrain[ cnnmodel3, images ];


Image@CZSample[ cnntrain3 ]


CZLogDensity[ cnntrain3, images[[1]] ]


<<"Experimental/GenerativeModels/CZPixelVaE.m"


cnnmodel2 = CZCreatePixelVaEBinaryImage[];


cnnvaetrain2 = CZTrain[ cnnmodel2, binImages ];


Image@CZSample[ cnnvaetrain2 ]


CZLogDensity[ cnnvaetrain2, binImages[[1]] ]


cnnvaemodel3 = CZCreatePixelVaEDiscreteImage[];


cnnvaetrain3 = CZTrain[ cnnvaemodel3, images ];


Image@CZSample[ cnnvaetrain3 ]


CZLogDensity[ cnnvaetrain3, images[[1]] ]
