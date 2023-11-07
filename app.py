from feature_extraction import feature_extraction
        
def main():
    extractor = feature_extraction()
    img_path = input("Entrer le chemin des images : ")
    feature, labels = extractor.feature_extraction(img_path)
    #print(feature)
    dataframe = extractor.transformation_dataframe(feature,labels)
    print(dataframe)
        
if __name__ == "__main__":
    main()