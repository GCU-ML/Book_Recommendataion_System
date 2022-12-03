# Overview

 &nbsp; As the people easily get information through media these days, book sales rate was getting decreased so we planned to make book recommendation system using three different method that can help improve book sales.

1. Item based collaborative filtering (KNN with cosine similarity)
2. Apriori algorithm
3. Content based filtering (Vector space model)

And we compared these three method to see the difference.

<br/>

# Function documentations

- Item based filtering
    
    ![Untitled_0](README_image/Untitled_0.png)
    
    ![Untitled_1](README_image/Untitled_1.png)
    
- **Apriori**
    
    ![Untitled_2](README_image/Untitled_2.png)
    
- Content based filtering
    
    ![Untitled_3](README_image/Untitled_3.png)
    
<br/>

# Analysis

 &nbsp; After making our system we compared the result of these three different methods. We gave same book name as input for apriori and content based method. For item based method we gave user id who read the book that we input on apriori and content based.

The followings are the results.

- Apriori result
    
    ![Untitled_4](README_image/Untitled_4.png)
    
- content based filtering - Tfidvectorizer
    
    ![Untitled_5](README_image/Untitled_5.png)
    
- Item based filtering - KNN
    
    ![Untitled_6](README_image/Untitled_6.png)
    

&nbsp; As we choosed “Harry Potter and the Order of the Phoenix” as an input book. Our team's idea of an ideal outcome was to recommend other Harry Potter series. However except for the item based method, there was many unexpected books in the recommendation list.<br/>
&nbsp; Apriori showed 0 of Harry Potter series, content based method showed 2 and Item based showed 4. As our apriori algorithm calculate similarity of books by their categories it might just have recommended the same category books.<br/>
&nbsp; In case of content based it showed better result than apriori. Even though they both use category for calculation, as content based method calculates similarity with the frequency of the category using Tfindvectorzier there were little difference on results.<br/>
&nbsp; Lastly the item based method showed most ideal result. We guess this is because item based method uses rating dataset which has high possibility of datas that users rated similar rating on same series. By this factor same series might be easily catched as high similarity on computation.

&nbsp; If we evaluate our system by this one test result alone, we can think that item based method showed the best performance. But if we think more deeply, the books other methods recommended  might also be attractive to users. So our team ended up with decision that evaluating recommendation system is really hard and there is no absolute optimal recommendation system.