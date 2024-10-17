from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["When your application runs on a version of Android that is more recent than your `targetSdkVersion` specifies that it has been tested with, various compatibility modes kick in. This ensures that your application continues to work, but it may look out of place. For example, if the `targetSdkVersion` is less than 14, your app may get an option button in the UI.\n\nTo fix this issue, set the `targetSdkVersion` to the highest available value. Then test your app to make sure everything works correctly. You may want to consult the compatibility notes to see what changes apply to each version you are adding support for: https://developer.android.com/reference/android/os/Build.VERSION_CODES.html as well as follow this guide:\nhttps://developer.android.com/distribute/best-practices/develop/target-sdk.html", 
             "This detector looks for usage of the Android Gradle Plugin where the version you are using is not the current stable release. Using older versions is fine, and there are cases where you deliberately want to stick with an older version. However, you may simply not be aware that a more recent version is available, and that is what this lint check helps find."]
embedding = model.encode(sentences)
print(f"Embedding:{embedding}")
# print(embedding.shape)