(* ::Package:: *)

<<"Experimental/GenerativeModels/CZNBModels.m"


mnist=ResourceData["MNIST","TrainingData"][[;;500,1]];


binImages = ImageData/@Binarize/@mnist;


images = CZDiscretize/@mnist;


nbmodel1 = CZCreateNBModel[];


nbtrain1 = CZTrain[ nbmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain1, binImages[[1]] ]


Image@CZSample[ nbtrain1 ]


nbmodel2 = CZCreateNBModel[ CZDiscrete[{28,28}] ];


nbtrain2 = CZTrain[ nbmodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain2, images[[1]] ]


Image@CZSample[ nbtrain2 ]


nbmodel3 = CZCreateNBModel[ CZRealGauss[{28,28}] ];


nbtrain3 = CZTrain[ nbmodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain3, ImageData@mnist[[1]] ]


Image@CZSample[ nbtrain3 ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


vaemodel1 = CZCreateVaE[];


vaetrain1 = CZTrain[ vaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain1, binImages[[1]] ]


Image@CZSample[ vaetrain1 ]


vaemodel2 = CZCreateVaE[ CZDiscrete[{28,28}] ];


vaetrain2 = CZTrain[ vaemodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain2, images[[1]] ]


Image@CZSample[ vaetrain2 ]


vaemodel3 = CZCreateVaE[ CZRealGauss[{28,28}] ];


vaetrain3 = CZTrain[ vaemodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain3, ImageData@mnist[[1]] ]


Image@CZSample[ vaetrain3 ]


<<"Experimental/GenerativeModels/CZPixelCNN.m"


cnnmodel1 = CZCreatePixelCNN[];


cnntrain1 = CZTrain[ cnnmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain1, binImages[[1]] ]


Image@CZSample[ cnntrain1 ]


cnnmodel2 = CZCreatePixelCNN[ CZDiscrete[{28,28}] ];


cnntrain2 = CZTrain[ cnnmodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain2, images[[1]] ]


Image@CZSample[ cnntrain2 ]


cnnmodel3 = CZCreatePixelCNN[ CZRealGauss[{28,28}] ];


cnntrain3 = CZTrain[ cnnmodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain3, ImageData@mnist[[1]] ]


Image@CZSample[ cnntrain3 ]


<<"Experimental/GenerativeModels/CZPixelVaE.m"


pixelvaemodel1 = CZCreatePixelVaEBinary[];


pixelvaetrain1 = CZTrain[ pixelvaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain1, binImages[[1]] ]


Image@CZSample[ pixelvaetrain1 ]


pixelvaemodel2 = CZCreatePixelVaEDiscrete[];


pixelvaetrain2 = CZTrain[ pixelvaemodel2, images, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain2, images[[1]] ]


Image@CZSample[ pixelvaetrain2 ]


pixelvaemodel3 = CZCreatePixelVaERealGauss[];


pixelvaetrain3 = CZTrain[ pixelvaemodel3, ImageData/@mnist, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain3, ImageData@mnist[[1]] ]


Image@CZSample[ pixelvaetrain3 ]
