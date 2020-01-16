(* ::Package:: *)

<<"Experimental/GenerativeModels/CZNBModels.m"


mnist=ResourceData["MNIST","TrainingData"][[;;500,1]];


binImages = ImageData/@Binarize/@mnist;


images = CZDiscretize/@mnist;


nbmodel1 = CZCreateNBModelBinary[];


nbtrain1 = CZTrain[ nbmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain1, binImages[[1]] ]


Image@CZSample[ nbtrain1 ]


nbmodel2 = CZCreateNBModelDiscrete[];


nbtrain2 = CZTrain[ nbmodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain2, images[[1]] ]


Image@CZSample[ nbtrain2 ]


nbmodel3 = CZCreateNBModelRealGauss[];


nbtrain3 = CZTrain[ nbmodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain3, ImageData@mnist[[1]] ]


Image@CZSample[ nbtrain3 ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


vaemodel1 = CZCreateVaEBinary[];


vaetrain1 = CZTrain[ vaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain1, binImages[[1]] ]


Image@CZSample[ vaetrain1 ]


vaemodel2 = CZCreateVaEDiscrete[];


vaetrain2 = CZTrain[ vaemodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain2, images[[1]] ]


Image@CZSample[ vaetrain2 ]


vaemodel3 = CZCreateVaERealGauss[];


vaetrain3 = CZTrain[ vaemodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain3, ImageData@mnist[[1]] ]


Image@CZSample[ vaetrain3 ]


<<"Experimental/GenerativeModels/CZPixelCNN.m"


cnnmodel1 = CZCreatePixelCNNBinary[];


cnntrain1 = CZTrain[ cnnmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain1, binImages[[1]] ]


Image@CZSample[ cnntrain1 ]


cnnmodel2 = CZCreatePixelCNNDiscrete[];


cnntrain2 = CZTrain[ cnnmodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain2, images[[1]] ]


Image@CZSample[ cnntrain2 ]


cnnmodel3 = CZCreatePixelCNNRealGauss[];


cnntrain3 = CZTrain[ cnnmodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain3, ImageData@mnist[[1]] ]


Image@CZSample[ cnntrain3 ]


<<"Experimental/GenerativeModels/CZPixelVaE.m"


cnnvaemodel1 = CZCreatePixelVaEBinary[];


cnnvaetrain1 = CZTrain[ cnnvaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ cnnvaetrain1, binImages[[1]] ]


Image@CZSample[ cnnvaetrain1 ]


cnnvaemodel2 = CZCreatePixelVaEDiscrete[];


cnnvaetrain2 = CZTrain[ cnnvaemodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ cnnvaetrain2, images[[1]] ]


Image@CZSample[ cnnvaetrain2 ]


cnnvaemodel3 = CZCreatePixelVaERealGauss[];


cnnvaetrain3 = CZTrain[ cnnvaemodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ cnnvaetrain3, ImageData@mnist[[1]] ]


Image@CZSample[ cnnvaetrain3 ]
