from feature_extraction import feature_extraction
        
def main():
    extractor = feature_extraction()
    img_path = input("Entrer le chemin des images : ")
    feature = extractor.feature_extraction(img_path)
    print(feature)
        
if __name__ == "__main__":
    main()