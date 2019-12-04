(* ::Package:: *)

<<"Experimental/GenerativeModels/CZNBModels.m"


mnist=ResourceData["MNIST","TrainingData"][[;;500,1]];


binVectors = Flatten/@ImageData/@Binarize/@mnist;


binImages = ImageData/@Binarize/@mnist;


images = CZDiscretize/@mnist;


nbmodel1 = CZCreateNBModelBinaryVector[];


nbtrain1 = CZTrain[ nbmodel1, binVectors ];


CZSample[ nbtrain1 ]


CZLogDensity[ nbtrain1, binVectors[[1]] ]


nbmodel2 = CZCreateNBModelBinaryImage[];


nbtrain2 = CZTrain[ nbmodel2, binImages ];


CZSample[ nbtrain2 ]


CZLogDensity[ nbtrain2, binImages[[1]] ]


nbmodel3 = CZCreateNBModelDiscreteImage[];


nbtrain3 = CZTrain[ nbmodel3, images ];


CZSample[ nbtrain3 ]


CZLogDensity[ nbtrain3, images[[1]] ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


vaemodel1 = CZCreateVaEBinaryVector[];


vaetrain1 = CZTrain[ vaemodel1, binVectors ];


CZSample[ vaetrain1 ]


CZLogDensity[ vaetrain1, binVectors[[1]] ]


vaemodel2 = CZCreateVaEBinaryImage[];


vaetrain2 = CZTrain[ vaemodel2, binImages ];


CZSample[ vaetrain2 ]


CZLogDensity[ vaetrain2, binImages[[1]] ]


vaemodel3 = CZCreateVaEDiscreteImage[];


vaetrain3 = CZTrain[ vaemodel3, images ];


CZSample[ vaetrain3 ]


CZLogDensity[ vaetrain3, images[[1]] ]
