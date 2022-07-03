# -*- coding: utf-8 -*-

__version__ = '1.0.1'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2017-May-25'
__url__     = 'https://engineering.purdue.edu/kak/distLSH/LocalitySensitiveHashing-1.0.1.html'
__copyright__ = "(C) 2017 Avinash Kak. Python Software Foundation."

__doc__ = '''

LocalitySensitiveHashing.py

Version: '''+ __version__ + '''

Author: Avinash Kak (kak@purdue.edu)

Date: '''+ __date__ + '''


@title
CHANGES:

  Version 1.0.1:

    This version fixes the typos and other errors discovered in the
    documentation.  The module code remains unchanged.
        

@title
INTRODUCTION:

    The LocalitySensitiveHashing module is an implementation of the
    Locality Sensitive Hashing (LSH) algorithm for nearest neighbor search.
    The main idea in LSH is to avoid having to compare every pair of data
    samples in a large dataset in order to find the nearest similar
    neighbors for the different data samples.  With LSH, one can expect a
    data sample and its closest similar neighbors to be hashed into the
    same bucket with a high probability.  By treating the data samples
    placed in the same bucket as candidates for similarity checking, we
    significantly reduce the computational burden associated with finding
    nearest neighbors in large datasets.
 
    While LSH algorithms have traditionally been used for finding nearest
    neighbors, this module goes a step further and explores using LSH for
    clustering the data.  Strictly speaking, this violates the basic
    mandate of LSH, which is to return just the nearest neighbors. (A data
    sample X being Y's nearest neighbor and Y being Z's nearest neighbor,
    in the sense nearest neighbors are commonly defined with the Cosine
    metric in LSH, does not always imply that X and Z will be sufficiently
    close to be considered each other's nearest neighbors.)  Nonetheless,
    if you believe that your datafile consists of non-overlapping data
    clusters, this module may do a decent job of finding those clusters.

    The rest of this section reviews the basic notions in LSH to help the
    user of this module understand the implementation code.  As to how
    these notions are used for clustering will be taken up in the next
    section.

    A hash function is locality sensitive if it places similar samples in
    the same bucket with a high probability, and if it places dissimilar
    samples in the same bucket with a low probability.  Two data samples
    are considered to be similar if the "distance" between them is at most
    d1 and two data samples are considered dissimilar if the "distance"
    between them is at least d2 = c * d1, with c > 1. (As for how to
    measure this "distance", for numerical data in a vector space, we have
    several distance measures at our disposal.  For example, we could use
    the Cosine distances or the L2-norm, etc.)  For given d1 and d2, the
    quality of a locality-sensitive hash function is measured by two
    probabilities, p1 and p2, where the former is the the least probability
    of collision for similar samples (which is something that we want) and
    the latter the largest probability of collision for dissimilar samples
    (which is something that we do NOT want).  For obvious reasons, you'd
    want p1 to be as high as possible and p2 to be as low as possible.  The
    locality sensitivity of such a hash function is characterized by the
    4-tuple (d1,d2,p1,p2).  We may refer to p1 as the least probability for
    detecting the true positives and p2 as largest probability of
    encountering false positives.
    
    In practice, it is not possible to come up with a single hash function
    with acceptably high true-positive probability and acceptably low
    false-positive probability.  However, it has been shown that a large
    number of hash functions working together in an AND-OR structure can
    give us the desired values for p1 and p2.  One starts out with a basic
    hash function that places similar samples in the same bucket with a
    high probability, but that, at the same time, places any two dissimilar
    samples in the same bucket with NOT a sufficiently low probability.
    Subsequently, one can require that for any two given samples to be
    considered candidates for similarity, they must be in the same bucket
    for a set of r random variants of the same basic hash function.  This
    is referred to as enforcing an ÁND over the r hash functions in order
    to decrease the collisions for dissimilar samples.  Since the ÁND
    operation can also diminish the probability of collisions for similar
    samples, we introduce an OR operation, which consists of ORing over b
    bands of ANDed operations.  Before visiting the collision probabilities
    for similar and dissimilar samples associated with this AND-OR logic,
    let me further illustrate in the next paragraph as to what is meant by
    the AND and the OR operations.  I'll do so by alluding to the basic
    data structure, htable_rows, that the module uses for the AND and the
    OR operations.
    
    This module implements the AND-OR idea described above in the following
    manner: The module constructs a 2D bit array, represented by
    htable_rows in the code, in which each row corresponds to a different
    hash function and each column corresponds to a data sample.  The total
    number of rows in htable_rows is b*r where r is the number of hash
    functions over which we want to carry out the AND operation and b is
    the number of such bands of r hash functions for the OR operation.  To
    elaborate, assume that both r and b are set to 3 and that we have a
    total of 5 data samples. The 2D array of 1's and 0's in htable_rows
    could look like what is shown below:
    
                                data samples 
         
                       x1     x2     x3     x4     x5
                     ----------------------------------
                    |                                             
                h1  |  1      .      1      .      .        b=0  r=0
                h2  |  0      .      0      .      .        b=0  r=1
    hash        h3  |  1      .      1      .      .        b=0  r=2
    functions       |
                h4  |  .      1      .      1      .        b=1  r=0
                h5  |  .      1      .      1      .        b=1  r=1 
                h6  |  .      0      .      0      .        b=1  r=2
                    |
                h7  |  1      .      .      .      1        b=2  r=0
                h8  |  1      .      .      .      1        b=2  r=1
                h9  |  1      .      .      .      1        b=2  r=2
                    |
    
    where an entry of "1" means that the hyperplane hash function placed
    that data sample in the "plus" bucket and an entry of "0" means that
    the data sample was placed in the "minus" bucket.  When the entry is a
    dot ".", that means we don't care about that entry for the sake of the
    explanation here. (Showing 1s and 0s for all entries would create too
    much visual clutter in the table.) Note also that, on the right of the
    table, I have taken the liberty of using the symbol b as the index
    vis-a-vis the parameter b, and the symbol r as the index vis-a-vis the
    parameter r.  Each distinct value for the index b shows a separate band
    of the hash functions.  And each distinct value for the index r for
    each index b shows the output of a separate hash function within each
    band.
            
    The AND-OR property says that for any two data samples to be considered
    similar they must agree with respect to all the hash values in ALL of
    the r rows IN AT LEAST ONE of the b bands.  (Now I am using b and r as
    the parameters of the algorithm.  In the example shown above, both b
    and r are set to 3.)  Based on this property, we claim that we have the
    following candidate pairs that we should test for similarity: x1 and
    x3, x2 and x4, and x1 and x5.
    
    Let's now revisit the question of what probabilities to associate with
    the collisions for similar and dissimilar pairs of samples through the
    AND-OR construction shown above.
    
    If p1 is the probability that two similar samples would be placed in
    the same bucket by the basic hash function used, the probability that r
    such functions would do the same in an AND aggregation of the hash
    functions is p1^r.  Along the same lines, if p2 is the probability that
    the basic hash function would place two dissimilar samples in the same
    bucket, then the probability of r such hash functions doing the same in
    an AND aggregation of the hash functions goes down to p2^r.  So, with
    regard to any single set of r hash functions, we have a desirable
    reduction in the collision probability associated with two dissimilar
    samples, but, unfortunately, also an undesirable reduction in the
    collision probability associated with two similar samples.  However,
    assuming that p1 is quite high to begin with, say around 0.9, and p2 is
    quite low to begin with, say, around 0.1. The p1^r value will be
    impacted much less that the p2^r value. So if the basic hash function
    is characterized by the 4-tuple (d1, d2, p1, p2) for its locality
    sensitivity, an r-wise AND aggregation of random variations on the
    basic hash function will be characterized by (d1, d2, p1^r,p2^r).
    
    Let's now see what OR logic does to a set of random variants of a (d1,
    d2, p1, p2) family of hash function. Let's say we are ORing over the
    bucket entries as produced by a family of b hash functions.  Since p1
    is the probability of collision for similar samples, we can say that
    each of the b hash functions would declare two similar samples to be
    dissimilar with a probability of (1 - p1).  That implies that, given a
    set of b such hash functions, none of them would declare two similar
    samples to be similar with a probability of (1 - p1)^b.  Therefore, the
    probability that at least one of these b hash functions in an OR
    construction would declare two similar samples to be similar is [1 - (1
    - p1)^b].  The same logic tells us that an OR combination over b hash
    functions would declare two dissimilar samples to be similar with a
    probability of [1 - (1 - p2)^b].  Hence, we can characterize the
    locality sensitivity of a set of ORed b hash functions by the 4-tuple
    (d1, d2, [1 - (1 - p1)^b], [1 - (1 - p2)^b]).
    
    It follows that if we apply the OR logic to b bands of hash functions,
    with each band consisting of r hash function operating together
    according to the AND logic, we can characterize the locality
    sensitivity of the b*r hash functions by the 4-tuple
    
           (d1, d2, [1 - (1 - p1^r)^b], [1 - (1 - p2^r)^b]).

    which we may express as (d1, d2, P1, P2) with P1 = 1 - (1 - p1^r)^b and
    P2 = 1 - (1 - p2^r)^b.  We can think of P1 and P2 as our amplified
    true-positive and false-positive probabilities.  Given target values
    for P1 and P2, one would need to solve the nonlinear equations that
    relate P1 and P2 with b and r for the values one should use for these
    two parameters in an LSH implementation.
    
    But what about the relationship between the distances d1 and d2, on the
    one hand, and the collision probabilities p1 and p2?  That depends on
    the choice of the basic hash function.  Consider what's surely the most
    commonly used hash function for finding nearest neighbors in
    multi-dimensional numerical data: the hyperplane hash function. Let's
    say your data consists of vectors in an N dimensional space.  For a
    hyperplane passing through the origin of the vector space, the
    orientation of the hyperplane being given by its surface normal, we can
    measure the similarity two data samples by finding out whether or not
    their projection on the normal to the hyperplane is on the same side of
    the hyperplane.  Let's say that the angle between the two data vectors
    is d1 (the angle being measured in the plane that passes through both
    the vectors).  We now say that for a randomly oriented hyperplane, the
    probability that the projections of the two data vectors on the normal
    to the hyperplane would fall on the same side of the hyperplane is (180
    - d) / 180.  Obviously, as d approaches zero, this probability will
    approach 1, which makes sense since when d=o, the projections of the
    two vectors on the surface normal will always be on the same side of
    the hyperplane for all its orientations.  On the other hand, when d
    approaches 180 degrees, you have two data vectors that are as
    dissimilar as they can be and in this case their projections on the
    surface normal will be on the opposite sides of the hyperplane for all
    its orientations.  For a non-zero angle d between the data vectors, as
    you consider hyperplanes with different possible random orientations
    that vary from 0 degrees to 180 degrees (each orientation being the
    angle of the surface normal in the plane containing both data vectors),
    over the orientation range 0 through 180-d degrees, both data vectors
    would project onto the normal on the same side of the hyperplane.
    These arguments apply to the relationship between d1 and p1 and also to
    the relationship between d2 and p2.  So we write:
    
                      180 - d1
             p1  =   ----------
                        180
    
                      180 - d2
             p2  =   ----------
                        180

    One can plug these formulas in the 4-tuple sensitivity characterization

           (d1, d2, [1 - (1 - p1^r)^b], [1 - (1 - p2^r)^b]).

    and given the desired targets for the probabilities for true-positive
    collisions and false-positive collisions, solve for the best values to
    use for b and r.

    For further information regarding Locality Sensitive Hash Functions,
    the reader is referred to Chapter 3 of the book "Mining of Massive
    Datasets" by Jure Leskovec, Anand Rajaraman, and Jeffery Ullman.
    

@title
CAN NEAREST NEIGHBORS RETURNED BY LSH BE USED TO CLUSTER THE DATA?

    Strictly speaking, the answer is no.  
    
    The problem is that the nearest neighbor property as calculated by LSH
    with hyperplane hash functions is not transitive.  Let's say that LSH
    considers data samples X and Y to be similar because the algorithm has
    placed them in the same bucket.  And, again, let's say that LSH
    considers the data samples Y and Z to be similar because the algorithm
    has placed them also in the same bucket (this bucket being different
    from the first bucket).  

    X being similar to Y and Y being similar to Z in the sense described
    here may not always imply that X is similar to Z.  Just imagine three
    points on the same great circle on the surface of a sphere.  The
    angular interval between X and Y, on the one hand, and between Y and Z,
    on the other, could be sufficiently small so that X and Y would be
    considered similar and Y and Z would be considered similar also.  Yet,
    the angular interval between X and Z may exceed the threshold test for
    similarity.
    
    Despite the difficulty created by the non-transitivity of angle-based
    measures of similarity between any two data samples, it may be possible
    to get good clustering results if the data is known to reside in
    non-overlapping clusters in the vector space. Although such a property
    is not likely to be true for most practical applications, modifications
    of the logic presented in this module may provide acceptable results
    when the data clusters are somewhat overlapping.  Consider, for
    example, the notion used in this module that two neighborhoods returned
    by LSH can be clustered together if they share data samples.  We could
    additionally predicate the joining of such neighborhoods on the
    distance between their means, or the distance between the shared data
    samples and the means of the two neighborhoods, with the idea that the
    shared data samples closer to the respective means in each of the
    neighborhoods are to be trusted more with regard to the two
    neighborhoods belonging to the same cluster.  This additional logic is
    not yet in the module.
    
    Think of this module as just a first step in the direction of exploring
    how one might exploit the neighborhood information returned by LSH in
    forming similarity clusters in data.
    

@title
USAGE:

    If you want to find just the nearest neighbors of a given data sample
    in a datafile that was processed by this module, your usage of the
    module will look like:

        lsh = LocalitySensitiveHashing( 
                   datafile = "data_for_lsh.csv",
                   dim = 10,
                   r = 50,         
                   b = 100,          
              )
        lsh.get_data_from_csv()
        lsh.initialize_hash_store()
        lsh.hash_all_data()
        similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()

    where 'dim' is the dimensionality of the numerical data in the file
    that in the above example is named "data_for_lsh.csv".  The datafile
    must be in the CSV format. (See the Examples directory for what a
    datafile must look like.)  The parameter 'r' is the number of rows for
    r-wise AND in each band of the hash functions, and the paramter 'b' is
    the number of bands for b-wise OR over all the bands.

    On the other hand, if you want to use this module for clustering your
    data using LSH for discovering the neighbors of the individual data
    samples, your usage of the module will look like:

        lsh = LocalitySensitiveHashing( 
                   datafile = "data_for_lsh.csv",
                   dim = 10,
                   r = 50,            
                   b = 100,              
                   expected_num_of_clusters = 10,
              )
        lsh.get_data_from_csv()
        lsh.initialize_hash_store()
        lsh.hash_all_data()
        similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
        coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
        merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )
        lsh.write_clusters_to_file( merged_similarity_groups, "clusters.txt" )
        
   where the constructor parameters 'dim', 'r', and 'b' carry the same
   meaning as mentioned for the previous case.  The new constructor
   parameter 'expected_num_of_clusters' specifies how many clusters you
   expect to see in the data.  The last four statements are about
   constructing clusters from the neighborhoods returned by LSH.  Their
   roles are explained elsewhere on this documentation page.  The next to
   the last statement shown above first orders the clusters formed
   according to their size.  It then retains a certain number of the
   largest clusters, the number being as specified by the constructor
   parameter expected_num_of_clusters.  The samples in the excess clusters
   are then pooled together and each sample in the pool assigned to the
   closest retained cluster.

   In the usage example shown above for clustering the data, you can
   replace the call in the next to the last statement by

        merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_set_based( coalesced_similarity_groups )

   This invokes a set based approach to merging the excess clusters with
   the retained clusters.  In the set based approach, you find the
   difference between the means of the excess clusters and the retained
   clusters.  An excess cluster is merged with a retained cluster if the
   difference between the means of the two is the least.


@title
CONSTRUCTOR PARAMETERS:

    datafile:            Must be a ".csv" file.  Each record in this file 
                         corresponds one data point in a vector space. 
                         Each record must have associated with it a 
                         unique symbolic name that must be in the first
                         column.

    dim:                 Is set to the dimensionality of the vector space
                         in which the data is defined.

    r:                   The number of rows in each band of the hash 
                         functions.  An r-wise AND operator is applied to 
                         the buckets in each such band.

    b:                   The number of r-row bands of the hash functions.  
                         A b-wise OR operator is applied to the buckets 
                         that correspond to each of the bands.

    expected_num_of_clusters:  This tell the module how many clusters
                         you expect to see in your datafile.

    
@title
METHODS:

    (1)  display_contents_of_all_hash_bins_pre_lsh()

         As mentioned elsewhere in this documentation page, each data
         sample is hashed by a randomly oriented hyperplane (passing
         through the origin of the vector space in which the data resides)
         by projecting the sample on the normal to the hyperplane. This
         creates a two-bin hash table, with some samples projected into the
         positive half-space and the others into the negative half-space.
         This method displays the orientation of the hyperplane, along with
         two lists, one consisting of the sample names that projected into
         the positive half-space and the other consisting of the sample
         names that projected into the negative half-space.

    (2)  evaluate_quality_of_similarity_groups( merged_similarity_groups )

         If the symbolic names in your datafile are based on the
         "sampleX_Y" format, where X is the integer ID of a cluster and Y
         the integer ID of a data element in the cluster, you can call on
         this method to evaluate the quality of the clusters produced by
         the LSH module.

    (3)  get_data_from_csv()

         This method extracts the numerical data from your CSV file.

    (4)  hash_all_data()

         It is this method that hashes all of your data records in the CSV
         file with r * b number of hash functions, each hash function being
         a randomly oriented hyperplane passing through the origin of the
         vector space in which the data resides.

    (5)  initialize_hash_store()

         This method must be called before the 'hash_all_data()' method.
         The initialization consists of generating the desired number of
         hyperplane orientations randomly and associating with each
         orientation a two-bin hash table in the form of a dictionary with
         two <key,value> pairs in it for the keys 'plus' and 'minus', with
         'plus' standing for the positive half-space and 'minus' for the
         negative half-space for each hyperplane.

    (6)  lsh_basic_for_nearest_neighbors()

         This method is this module's implementation of the hyperplane
         based LSH algorithm.  This method's mandate is what is
         traditionally accomplished with LSH --- finding nearest neighbors
         for data elements.

    (7)  lsh_basic_for_neighborhood_clusters()

         This method is a slight variation on the previous method, in that,
         instead of returning the nearest neighbors of a data element, it
         merges the data element with its LSH-discovered neighbors to form
         a cluster.  The method returns a list of such clusters.

    (8)  merge_similarity_groups_with_coalescence()

         This is where we violate the traditional mandate of the LSH
         algorithm --- so use it with care.  Make doubly sure that this and
         other similar methods are appropriate for your data.  This method
         merges together those clusters produced by the previous method if
         they share any data elements.  As you can well imagine, if the
         data is noisy and the clusters in the data are overlapping, this
         method may cluster all of your data into a single large cluster.

    (9)  merge_similarity_groups_with_l2norm_sample_based()

         A sign of the coalescence step in the previous method working
         reasonably well is that the number of clusters produced by the
         previous method will be larger than the number of clusters
         actually present in the data.  If that is the case, you can use
         this method to merge the excess clusters with those that are
         retained.  This method first orders by size the clusters produced
         by the previous method.  Subsequently, it pools together all the
         samples in the excess clusters.  Each sample in this pool is then
         merged with one of the retained clusters on the basis of the least
         distance between the sample and cluster mean.

    (10) merge_similarity_groups_with_l2norm_set_based()

         The contract of this method is the same as that of the previous
         method, except for the difference that the excess clusters are
         merged wholesale with the retained clusters.  The methods computes
         the difference in the mean vectors between a given excess cluster
         and each of the retained clusters. An excess cluster is merged
         with the nearest retained cluster on the basis of this difference
         in the means of the two being the smallest.

    (11) prune_similarity_groups(self):

         If the module is producing too many small clusters and you don't
         see any value in retaining them, you can call on this method to
         get rid of them.  In order to use this method, you must first set
         the constructor parameter 'similarity_group_min_size_threshold' to
         the minimum cluster size you want.

    (12) show_data_for_lsh()
 
         For the purpose of verification, this method shows you what data
         was extracted from the CSV file.

    (13) write_clusters_to_file()    

         You can call on this method to write the clusters out to a disk
         file.  The method takes two arguments, the first for the list of
         clusters returned by any of the methods in items (8), (9), and
         (10) above, and the second for the name of the disk file.

@title
The DataGenerator CLASS:

    The module comes equipped with a DataGenerator class that you can use
    to generate multi-class multi-variate data for experimenting with the
    LSH module.

    The DataGenerator class is programmed to generate N "balls" of
    multi-variate Gaussian data, where N is the value for the parameter
    'how_many_similarity_groups' shown in the constructor call
    below. Consider an N dimensional cube in the positive quadrant of an
    N-dimensional space.  Such a cube has 2^N vertices. The N Gaussian
    balls are centered at the N vertices of the cube that are closest to
    the origin.

        import LocalitySensitiveHashing
        dim = 10
        covar = numpy.diag([0.01] * dim)
        output_file = 'data_for_lsh.csv'
        data_gen = DataGenerator(
                                  output_csv_file   = output_file,
                                  how_many_similarity_groups = 10,
                                  dim = dim,
                                  number_of_samples_per_group = 8,
                                  covariance = covar,
                                )
        
        data_gen.gen_data_and_write_to_csv()
        

@title
THE EXAMPLES DIRECTORY:

    The best way to become familiar with this module is by executing the
    following scripts in the Examples subdirectory:

    1.  LSH_basic_for_demonstrating_nearest_neighbors.py

            This script demonstrates the functionality one traditionally
            associates with the LSH algorithm --- finding nearest
            neighbors.  This script places the user in an interactive
            session in which the user is asked to enter the symbolic name
            of a data record in the datafile that was processed by the LSH
            algorithm.  Subsequently, the user is shown the nearest
            neighbors of that record.

    2.  Clustering_with_LSH_with_sample_based_merging.py

            This is one of the two scripts in the Examples directory that
            attempt to cluster the data that is supplied to the module
            through a CSV file. The data is clustered on the basis of the
            neighborhoods supplied by the LSH algorithm.  If the
            coalescence of the LSH generated neighborhood creates more
            cluster than expected, this script calls on the
            merge_similarity_groups_with_l2norm_sample_based() to merge
            excess small clusters with the main retained clusters.

    3.  Clustering_with_LSH_with_set_based_merging.py

            This is the second of the cluster producing scripts in the
            Examples directory.  Again, the starting points for forming the
            clusters are the neighborhoods supplied by the LSH algorithm.
            Unlike the previous script, this script calls on the
            merge_similarity_groups_with_l2norm_set_based() to merge excess
            small clusters with the main retained clusters.

    4.  gen_data.py

            This script uses the DataGenerator class that comes with the
            LSH module to generate multi-variate Gaussian data for
            experimenting with the LSH module.  The output file produced by
            this method can be used directly for input to the LSH class.


@title
CAVEATS:

    Assuming that your data contains non-overlapping clusters, as to what
    sort of clustering results you'll get with this module depends a great
    deal on your choice of b and r parameters.  If you can make a good
    guess at the distances d1 and d2 appropriate for your application
    (recall from the Introduction that we consider two data elements to be
    similar if their angular difference in the vector space is at most d1;
    and we consider them to be dissimilar if their angular difference is at
    least d2 = c * d1 with c > 1), and if you can come up with the target
    true-positive collision probability P1 and false-positive collision
    probability P2 as defined in the Introduction, you can solve the
    nonlinear equations presented there for the best values to use for b
    and r.


@title  
INSTALLATION:
                                                                                                   
    The LSH class was packaged using setuptools.  For installation, execute
    the following command in the source directory (this is the directory
    that contains the setup.py file after you have downloaded and
    uncompressed the package):
                                                                                                   
            sudo python setup.py install                                                           
                                                                                                   
    and/or, for the case of Python 3,                                                               
                                                                                                   
            sudo python3 setup.py install                                                          
                                                                                                   
    On Linux distributions, this will install the module file at a location                        
    that looks like                                                                                
                                                                                                   
             /usr/local/lib/python2.7/dist-packages/                                               
                                                                                                   
    and, for the case of Python 3, at a location that looks like                                    
                                                                                                   
             /usr/local/lib/python3.4/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the LSH class:

        import sys
        sys.path.append( "pathname_to_LSH_directory" )

    To uninstall the module, simply delete the source directory, locate
    where LSH was installed with "locate LSH" and delete those files.  As
    mentioned above, the full pathname to the installed version is likely
    to look like /usr/local/lib/python2.7/dist-packages/LSH*

    If you want to carry out a non-standard install of the LSH module,
    look up the on-line information on Disutils by pointing your
    browser to

          http://docs.python.org/dist/dist.html

@title
ACKNOWLEDGMENTS:

    The author has learned much from his LSH-related discussions with Tommy
    Chang who is currently finishing his Ph.D. in the Robot Vision Lab at
    Purdue.  Tommy is using LSH to solve a truly big-data problem: the
    problem of creating concise training and testing datasets for wide-area
    land-cover classification algorithms that must work well on hundreds of
    satellite images (as opposed to just one satellite image at a time,
    which is the norm in the remote-sensing community).  Typically, these
    images cover a region of the earth whose size may be as large as
    hundreds of thousands of square kilometers.


@title
ABOUT THE AUTHOR:

    The author, Avinash Kak, recently finished his 17-year long Objects
    Trilogy project with the publication of the book "Designing with
    Objects" by John-Wiley. If interested, check out his web page at Purdue
    to find out what the Objects Trilogy project was all about. You might
    like "Designing with Objects" especially if you enjoyed reading Harry
    Potter as a kid (or even as an adult, for that matter).


@endofdocs
'''


import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *

#-----------------------------------  Utility Functions  ------------------------------------

def sample_index(sample_name):
    '''
    We assume that the raw data is stored in the following form:

       sample0_0,0.951,-0.134,-0.102,0.079,0.12,0.123,-0.03,-0.078,0.036,0.138
       sample0_1,1.041,0.057,0.095,0.026,-0.154,0.231,-0.074,0.005,0.055,0.14
       ...
       ...
       sample1_8,-0.153,1.083,0.041,0.086,-0.059,0.042,-0.172,0.014,-0.153,0.091
       sample1_9,0.051,1.122,-0.014,-0.117,0.015,-0.044,0.011,0.008,-0.121,-0.017
       ...
       ...

    This function returns the second integer in the name of each data record.
    It is useful for sorting the samples and for visualizing whether or not
    the final clustering step is working correctly.
    '''
    m = re.search(r'(.+)$', sample_name)
    return int(m.group(1))

def sample_group_index(sample_group_name):
    '''
    As the comment block for the previous function explains, the data sample
    for LSH are supposed to have a symbolic name at the beginning of the 
    comma separated string.  These symbolic names look like 'sample0_0', 
    'sample3_4', etc., where the first element of the name, such as 'sample0',
    indicates the group affiliation of the sample.  The purpose of this
    function is to return just the integer part of the group name.
    '''
    m = re.search(r'^.*(\d+)', sample_group_name)
    return int(m.group(1))

def band_hash_group_index(block_name):
    '''
    The keys of the final output that is stored in the hash self.coalesced_band_hash
    are strings that look like:

         "block3 10110"

    This function returns the block index, which is the integer that follows the 
    word "block" in the first substring in the string that you see above.
    '''
    firstitem = block_name.split()[0]
    m = re.search(r'(\d+)$', firstitem)
    return int(m.group(1))

def deep_copy_array(array_in):
    '''
    Meant only for an array of scalars (no nesting):
    '''
    array_out = []
    for i in range(len(array_in)):
        array_out.append( array_in[i] )
    return array_out

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def cleanup_csv(line):
    line = line.translate(bytes.maketrans(b":?/()[]{}'",b"          ")) \
           if sys.version_info[0] == 3 else line.translate(string.maketrans(":?/()[]{}'","          "))
    double_quoted = re.findall(r'"[^\"]*"', line[line.find(',') : ])         
    for item in double_quoted:
        clean = re.sub(r',', r'', item[1:-1].strip())
        parts = re.split(r'\s+', clean.strip())
        line = str.replace(line, item, '_'.join(parts))
    white_spaced = re.findall(r',(\s*[^,]+)(?=,|$)', line)
    for item in white_spaced:
        litem = item
        litem = re.sub(r'\s+', '_', litem)
        litem = re.sub(r'^\s*_|_\s*$', '', litem) 
        line = str.replace(line, "," + item, "," + litem) if line.endswith(item) else str.replace(line, "," + item + ",", "," + litem + ",") 
    fields = re.split(r',', line)
    newfields = []
    for field in fields:
        newfield = field.strip()
        if newfield == '':
            newfields.append('NA')
        else:
            newfields.append(newfield)
    line = ','.join(newfields)
    return line

# Needed for cleanly terminating the interactive method lsh_basic_for_nearest_neighbors():
def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)

#----------------------------------- LSH Class Definition ------------------------------------

class LocalitySensitiveHashing(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception(  
                   '''LocalitySensitiveHashing constructor can only be called with keyword arguments for the 
                      following keywords: datafile,csv_cleanup_needed,how_many_hashes,r,b,
                      similarity_group_min_size_threshold,debug,
                      similarity_group_merging_dist_threshold,expected_num_of_clusters''') 
        allowed_keys = 'datafile','dim','csv_cleanup_needed','how_many_hashes','r','b','similarity_group_min_size_threshold','similarity_group_merging_dist_threshold','expected_num_of_clusters','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        datafile=dim=debug=csv_cleanup_needed=how_many_hashes=r=b=similarity_group_min_size_threshold=None
        similarity_group_merging_dist_threshold=expected_num_of_clusters=None
        if kwargs and not args:
            if 'csv_cleanup_needed' in kwargs : csv_cleanup_needed = kwargs.pop('csv_cleanup_needed')
            if 'datafile' in kwargs : datafile = kwargs.pop('datafile')
            if 'dim' in kwargs :  dim = kwargs.pop('dim')
            if 'r' in kwargs  :  r = kwargs.pop('r')
            if 'b' in kwargs  :  b = kwargs.pop('b')
            if 'similarity_group_min_size_threshold' in kwargs  :  
                similarity_group_min_size_threshold = kwargs.pop('similarity_group_min_size_threshold')
            if 'similarity_group_merging_dist_threshold' in kwargs  :  
                similarity_group_merging_dist_threshold = kwargs.pop('similarity_group_merging_dist_threshold')
            if 'expected_num_of_clusters' in kwargs  :  
                expected_num_of_clusters = kwargs.pop('expected_num_of_clusters')
            if 'debug' in kwargs  :  debug = kwargs.pop('debug')
        # if datafile:
        #     self.datafile = datafile
        # else:
        #     raise Exception("You must supply a datafile")
        self._csv_cleanup_needed = csv_cleanup_needed
        self.similarity_group_min_size_threshold = similarity_group_min_size_threshold
        self.similarity_group_merging_dist_threshold = similarity_group_merging_dist_threshold
        self.expected_num_of_clusters = expected_num_of_clusters
        if dim:
            self.dim = dim
        else:
            raise Exception("You must supply a value for 'dim' which stand for data dimensionality")
        self.r = r                               # Number of rows in each band (each row is for one hash func)
        self.b = b                               # Number of bands.
        self.how_many_hashes =  r * b
        self._debug = debug
        self._data_dict = {}                     # sample_name =>  vector_of_floats extracted from CSV stored here
        self.how_many_data_samples = 0
        self.hash_store = {}                     # hyperplane =>  {'plus' => set(), 'minus'=> set()}
        self.htable_rows  = {}
        self.index_to_hplane_mapping = {}
        self.band_hash = {}                      # BitVector column =>  bucket for samples  (for the AND action)
        self.band_hash_mean_values = {}          # Store the mean of the bucket contents in band_hash dictionary
        self.similarity_group_mean_values = {}
        self.coalesced_band_hash = {}            # Coalesce those keys of self.band_hash that have data samples in common
        self.similarity_groups = []
        self.coalescence_merged_similarity_groups = []  # Is a list of sets
        self.l2norm_merged_similarity_groups = []  # Is a list of sets
        self.merged_similarity_groups = None
        self.pruned_similarity_groups = []
        self.evaluation_classes = {}             # Used for evaluation of clustering quality if data in particular format

    # def get_data_from_csv(self):
    #     if not self.datafile.endswith('.csv'): 
    #         Exception("Aborted. get_training_data_from_csv() is only for CSV files")
    #     data_dict = {}
    #     with open(self.datafile) as f:
    #         for i,line in enumerate(f):
    #             if line.startswith("#"): continue      
    #             record = cleanup_csv(line) if self._csv_cleanup_needed else line
    #             parts = record.rstrip().split(r',')
    #             data_dict[parts[0].strip('"')] = list(map(lambda x: convert(x), parts[1:]))
    #             if i%10000 == 0:
    #                 print('.'),
    #                 sys.stdout.flush()
    #             sys.stdout = sys.__stdout__
    #         f.close() 
    #     self.how_many_data_samples = i + 1
    #     self._data_dict = data_dict

    def get_data_from_nparray(self, input_array):
        if not isinstance(input_array, numpy.ndarray):
            Exception("Aborted. input must be a numpy array.")
        data_dict = {}
        for i in range(input_array.shape[0]):
            data_dict[str(i)] = list(input_array[i])
        self.how_many_data_samples = i + 1
        self._data_dict = data_dict

    def show_data_for_lsh(self):
        print("\n\nData Samples:\n\n")
        for item in sorted(self._data_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item)

    def initialize_hash_store(self):
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

    def hash_all_data_with_one_hyperplane(self):
        hyperplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
        print( "hyperplane: %s" % str(hyperplane) )
        hyperplane = hyperplane / numpy.linalg.norm(hyperplane)
        for sample in self._data_dict:
            bin_val = numpy.dot( hyperplane, self._data_dict[sample])
            bin_val = 1 if bin_val>= 0 else -1      
            print( "%s: %s" % (sample, str(bin_val)) )

    def hash_all_data(self):
        for hplane in self.hash_store:
            for sample in self._data_dict:
                hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  ")) \
                       if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
                bin_val = numpy.dot(list(map(convert, hplane_vals.split())), self._data_dict[sample])
                bin_val = 1 if bin_val>= 0 else -1      
                if bin_val>= 0:
                    self.hash_store[hplane]['plus'].add(sample)
                else:
                    self.hash_store[hplane]['minus'].add(sample)

    def hash_all_data_tf(self, data):
        import tensorflow as tf
        import numpy as np
        hplane_array = []
        hplane_list = []
        for hplane in self.hash_store:
            hplane_list.append(hplane)
            hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  ")) \
                   if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
            hplane_array.append(list(map(convert, hplane_vals.split())))
        hplane_array = np.array(hplane_array, dtype=np.float64)
        data.astype(np.float32)
        hplane_array.astype(np.float32)
        # print(hplane_array.shape)
        # print(self._data_dict)
        # print(hplane_array.shape, data.shape)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                bin_val = tf.matmul(hplane_array, data.T)
                bin_val = bin_val.eval(session=sess)
        # print(self.hash_store)
        # exit(0)
        # for i, hplane in zip(range(bin_val.shape[0]), hplane_list):
        for i, hplane in enumerate(hplane_list, 0):
            for j in range(bin_val.shape[1]):
                if bin_val[i][j] >= 0:
                    self.hash_store[hplane]['plus'].add(str(j))
                else:
                    self.hash_store[hplane]['minus'].add(str(j))


    def lsh_basic_for_nearest_neighbors(self):
        '''
        Regarding this implementation of LSH, note that each row of self.htable_rows corresponds to 
        one hash function.  So if you have 3000 hash functions for 3000 different randomly chosen 
        orientations of a hyperplane passing through the origin of the vector space in which the
        numerical data is defined, this table has 3000 rows.  Each column of self.htable_rows is for
        one data sample in the vector space.  So if you have 80 samples, then the table has 80 columns.
        The output of this method consists of an interactive session in which the user is asked to
        enter the symbolic name of a data record in the dataset processed by the LSH algorithm. The
        method then returns the names (some if not all) of the nearest neighbors of that data point.
        '''
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        ## Print BitVectors
        # for (i,_) in enumerate(sorted(self.hash_store)):
        #     if i % self.r == 0: print()
        #     print( str(self.htable_rows[i]) )
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):                
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)
        if self._debug:
            print( "\n\nPre-Coalescence results:" )
            for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):
                print()
                print( "%s    =>   %s" % (key, str(self.band_hash[key])) )
        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        sample_name_to_hash_key = {}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
                sample_name_to_hash_key[sample_name] = key
        # while True:
        #     sample_name = None
        #     if sys.version_info[0] == 3:
        #         sample_name =  input('''\nEnter the symbolic name for a data sample '''
        #                              '''(must match names used in your datafile): ''')
        #     else:
        #         sample_name = raw_input('''\nEnter the symbolic name for a data sample '''
        #                                 '''(must match names used in your datafile): ''')
        #     if sample_name in similarity_neighborhoods:
        #         print( "\nThe nearest neighbors of the sample: %s" % str(similarity_neighborhoods[sample_name]) )
        #     else:
        #         print( "\nThe name you entered does not match any names in the database.  Try again." )
        # for sample_name in similarity_neighborhoods:
        #     print("For ", sample_name, ", recording ", len(similarity_neighborhoods[sample_name]), "samples.")
        return similarity_neighborhoods, sample_name_to_hash_key

    def lsh_basic_for_neighborhood_clusters(self):
        '''
        This method is a variation on the method lsh_basic_for_nearest_neighbors() in the following
        sense: Whereas the previous method outputs a hash table whose keys are the data sample names
        and whose values are the immediate neighbors of the key sample names, this method merges
        the keys with the values to create neighborhood clusters.  These clusters are returned as 
        a list of similarity groups, with each group being a set.
        '''
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        for (i,_) in enumerate(sorted(self.hash_store)):
            if i % self.r == 0: print
            print( str(self.htable_rows[i]) ) 
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):                
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)
        if self._debug:
            print("\n\nPre-Coalescence results:")
            for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):
                print()
                print( "%s    =>    %s" % (key, str(self.band_hash[key])) )
        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        print("\n\nSimilarity neighborhoods calculated by the basic LSH algo:")
        for key in sorted(similarity_neighborhoods, key=lambda x: sample_index(x)):
            print( "\n  %s   =>  %s" % (key, str(sorted(similarity_neighborhoods[key], key=lambda x: sample_index(x)))) )
            simgroup = set(similarity_neighborhoods[key])
            simgroup.add(key)
            self.similarity_groups.append(simgroup)
        print( "\n\nSimilarity groups calculated by the basic LSH algo:\n" )
        for group in self.similarity_groups:
            print(str(group))
            print()
        print( "\nTotal number of similarity groups found by the basic LSH algo: %d" % len(self.similarity_groups) )
        return self.similarity_groups

    def merge_similarity_groups_with_coalescence(self, similarity_groups):
        '''
        The purpose of this method is to do something that, strictly speaking, is not the right thing to do
        with an implementation of LSH.  We take the clusters produced by the method 
        lsh_basic_for_neighborhood_clusters() and we coalesce them based on the basis of shared data samples.
        That is, if two neighborhood clusters represented by the sets A and B have any data elements in 
        common, we merge A and B by forming the union of the two sets.
        '''
        merged_similarity_groups = []
        for group in similarity_groups:
            if len(merged_similarity_groups) == 0:
                merged_similarity_groups.append(group)
            else:
                new_merged_similarity_groups = []
                merge_flag = 0
                for mgroup in merged_similarity_groups:
                    if len(set.intersection(group, mgroup)) > 0:
                        new_merged_similarity_groups.append(mgroup.union(group))
                        merge_flag = 1
                    else:
                       new_merged_similarity_groups.append(mgroup)
                if merge_flag == 0:
                    new_merged_similarity_groups.append(group)     
                merged_similarity_groups = list(map(set, new_merged_similarity_groups))
        for group in merged_similarity_groups:
            print( str(group) )
            print()
        print( "\n\nTotal number of MERGED similarity groups using coalescence: %d" % len(merged_similarity_groups) )
        self.coalescence_merged_similarity_groups = merged_similarity_groups
        return merged_similarity_groups

    def merge_similarity_groups_with_l2norm_sample_based(self, similarity_groups):
        '''
        The neighborhood set coalescence as carried out by the previous method will generally result
        in a clustering structure that is likely to have more clusters than you may be expecting to
        find in your data. This method first orders the clusters (called 'similarity groups') according 
        to their size.  It then pools together the data samples in the trailing excess similarity groups.  
        Subsequently, for each data sample in the pool, it merges that sample with the closest larger 
        group.
        '''
        similarity_group_mean_values = {}
        for group in similarity_groups:            #  A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
            if self._debug:
                print( "\n\nCLUSTER MEAN: %f" % group_mean )
        new_similarity_groups = []
        key_to_small_group_mapping = {}
        key_to_large_group_mapping = {}
        stringified_list = [str(item) for item in similarity_groups]
        small_group_pool_for_a_given_large_group = {x : [] for x in stringified_list}
        if len(similarity_groups) > self.expected_num_of_clusters:
            ordered_sim_groups_by_size = sorted(similarity_groups, key=lambda x: len(x), reverse=True)
            retained_similarity_groups = ordered_sim_groups_by_size[:self.expected_num_of_clusters]
            straggler_groups = ordered_sim_groups_by_size[self.expected_num_of_clusters :]
            print( "\n\nStraggler groups: %s" % str(straggler_groups) )
            samples_in_stragglers =  sum([list(group) for group in straggler_groups], [])
            print( "\n\nSamples in stragglers: %s" %  str(samples_in_stragglers) )
            straggler_sample_to_closest_retained_group_mapping = {sample : None for sample in samples_in_stragglers}
            for sample in samples_in_stragglers:
                dist_to_closest_retained_group_mean, closest_retained_group = None, None
                for group in retained_similarity_groups:
                        dist = l2norm(similarity_group_mean_values[str(group)], self._data_dict[sample])
                        if dist_to_closest_retained_group_mean is None:
                            dist_to_closest_retained_group_mean = dist
                            closest_retained_group = group
                        elif dist < dist_to_closest_retained_group_mean:
                            dist_to_closest_retained_group_mean = dist
                            closest_retained_group = group
                        else:
                            pass
                straggler_sample_to_closest_retained_group_mapping[sample] = closest_retained_group
            for sample in samples_in_stragglers:
                straggler_sample_to_closest_retained_group_mapping[sample].add(sample)
            print( "\n\nDisplaying sample based l2 norm merged similarity groups:" )
            self.merged_similarity_groups_with_l2norm = retained_similarity_groups
            for group in self.merged_similarity_groups_with_l2norm:
                print( str(group) )
            return self.merged_similarity_groups_with_l2norm
        else:
            print('''\n\nNo sample based merging carried out since the number of clusters yielded by coalescence '''
                  '''is fewer than the expected number of clusters.''')
            return similarity_groups

    def merge_similarity_groups_with_l2norm_set_based(self, similarity_groups):
        '''
        The overall goal of this method is the same as that of 
        merge_similarity_groups_with_l2norm_sample_based(), except for the difference that
        we now merge the excess similarity groups wholesale with the retained similarity 
        groups.  For each excess similarity group, we find the closest retained similarity group,
        closest in terms of the l2 norm distance between the mean values of the two groups.
        '''    
        similarity_group_mean_values = {}
        for group in similarity_groups:                # A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
            if self._debug:
                print( "\n\nCLUSTER MEAN: %f" % group_mean )
        if len(similarity_groups) > self.expected_num_of_clusters:
            new_similarity_groups = []
            key_to_small_group_mapping = {}
            key_to_large_group_mapping = {}
            ordered_sim_groups_by_size = sorted(similarity_groups, key=lambda x: len(x), reverse=True)
            retained_similarity_groups = ordered_sim_groups_by_size[:self.expected_num_of_clusters]
            straggler_groups = ordered_sim_groups_by_size[self.expected_num_of_clusters :]
            print( "\n\nStraggler groups: %s" % str(straggler_groups) )
            print( "\n\nNumber of samples in retained groups: %d" % len(list(set.union(*retained_similarity_groups))) )
            print( "\n\nNumber of samples in straggler groups: %d" % len(list(set.union(*straggler_groups))) )
            retained_stringified_list = [str(item) for item in retained_similarity_groups]
            small_group_pool_for_a_given_large_group = {x : [] for x in retained_stringified_list}
            for group1 in straggler_groups:
                key_to_small_group_mapping[str(group1)] = group1
                dist_to_closest_large_group_mean, closest_large_group = None, None
                for group2 in retained_similarity_groups:
                    key_to_large_group_mapping[str(group2)] = group2
                    dist = l2norm(similarity_group_mean_values[str(group2)], similarity_group_mean_values[str(group1)])
                    if dist_to_closest_large_group_mean is None:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    elif dist < dist_to_closest_large_group_mean:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    else:
                        pass
                small_group_pool_for_a_given_large_group[str(closest_large_group)].append(group1)
            if any(len(small_group_pool_for_a_given_large_group[x]) > 0 for x in small_group_pool_for_a_given_large_group):
                print( "\n\nTHERE IS NON-ZERO POOL FOR MERGING FOR AT LEAST ONE LARGER SIMILARITY GROUPS" )
                print( str(small_group_pool_for_a_given_large_group.values()) )
            for key in small_group_pool_for_a_given_large_group:
                lgroup = key_to_large_group_mapping[key]
                list_fo_small_groups = small_group_pool_for_a_given_large_group[key]
                print( "\n\nFor group %s, the pool of small groups for merging =====>  %s" % 
                                                                          (str(lgroup), str(list_fo_small_groups)) )
            for group in sorted(retained_similarity_groups, key=lambda x: len(x), reverse=True):
                group_copy = set(group)     # shallow copy
                if len(small_group_pool_for_a_given_large_group[str(group)]) > 0:
                    for setitem in small_group_pool_for_a_given_large_group[str(group)]:
                        group_copy.update(setitem)  
                    new_similarity_groups.append(group_copy)
                else:
                    new_similarity_groups.append(group_copy)
            self.merged_similarity_groups_with_l2norm = new_similarity_groups
            print( "\n\nDisplaying set based l2 norm merged similarity groups:")
            for group in new_similarity_groups:
                print( str(group) )
            return new_similarity_groups
        else:
            print('''\n\nNo set based merging carried out since the number of clusters yielded by coalescence '''
                  '''is fewer than the expected number of clusters.''')
            return similarity_groups

    def prune_similarity_groups(self):
        '''
        If your data produces too many similarity groups, you can get rid of the smallest with
        this method.  In order to use this method, you must specify a value for the parameter
        'similarity_group_min_size_threshold' in the call to the constructor of the LSH module.
        '''
        if self.merged_similarity_groups is not None:
            self.pruned_similarity_groups = [x for x in self.merged_similarity_groups if len(x) > 
                                                            self.similarity_group_min_size_threshold] 
        else:
            self.pruned_similarity_groups = [x for x in self.similarity_groups if len(x) > 
                                                        self.similarity_group_min_size_threshold]
        print( "\nNumber of similarity groups after pruning: %d" % len(self.pruned_similarity_groups) )      
        print( "\nPruned similarity groups: " )
        for group in self.pruned_similarity_groups:
            print( str(group) )
        return self.pruned_similarity_groups

    def evaluate_quality_of_similarity_groups(self, evaluation_similarity_groups):
        '''
        The argument to this method, evaluation_similarity_groups, is a list of sets, with each set being 
        a similarity group, which is the same thing as a cluster.

        If you plan to invoke this method to evaluate the quality of clustering achieved by the values
        used for the parameters r and b, you'd want the data records in the CSV datafile to look like:

            sample0_3,0.925,-0.008,0.009,0.058,0.092,0.117,-0.076,0.239,0.086,-0.149
        
        Note in particular the syntax used for naming a data record. The name 'sample0_3' means that this 
        is the 3rd sample generated randomly for data class 0.  The goal of this method is to example all 
        such  sample names and figure out how many classes exist in the data.
        '''
        print( '''\n\nWe measure the quality of a similarity group by taking stock of how many '''
               '''different different input similarity groups are in the same output similarity group.''')
        sample_classes = set()
        for item in sorted(self._data_dict.items(), key = lambda x: sample_index(x[0]) ):
            sample_classes.add(item[0][:item[0].find(r'_')])
        self.evaluation_classes = sample_classes
        if len(self.evaluation_classes) == 0:
            sys.exit('''\n\nUnable to figure out the number of data classes in the datafile processed by '''
                     '''this module --- aborting''')                     
        total_num_samples_in_all_similarity_groups = 0
        print( "\n\nTotal number of similarity groups tested: %d" % len(evaluation_similarity_groups) )
        m = re.search('^([a-zA-Z]+).+_', list(self._data_dict.keys())[0])
        sample_name_stem = m.group(1)
        for group in sorted(evaluation_similarity_groups, key=lambda x: len(x), reverse=True):
            total_num_samples_in_all_similarity_groups += len(group)
            set_for_sample_ids_in_group = set()
            how_many_uniques_in_each_group = {g : 0 for g in self.evaluation_classes}
            for sample_name in group:
                m = re.search('^[\w]*(.+)_', sample_name)
                group_index_for_sample = int(m.group(1))
                set_for_sample_ids_in_group.add(group_index_for_sample)
                how_many_uniques_in_each_group[sample_name_stem + str(group_index_for_sample)] += 1
            print( "\n\nSample group ID's in this similarity group: %s" % str(set_for_sample_ids_in_group) )
            print( "    Distribution of sample group ID's in similarity group: " )
            for key in sorted(how_many_uniques_in_each_group, key=lambda x: sample_group_index(x)):
                if how_many_uniques_in_each_group[key] > 0:
                    print("        Number of samples with Group ID " + str(sample_group_index(key)) + 
                                                           " => " + str(how_many_uniques_in_each_group[key]))
            if len(set_for_sample_ids_in_group) == 1:
                print("    Group purity level: ", 'pure')
            else:
                print("    Group purity level: ", 'impure')
        print( "\n\nTotal number of samples in the different clusters: %d" % total_num_samples_in_all_similarity_groups )

    def write_clusters_to_file(self, clusters, filename):
        FILEOUT = open(filename, 'w')
        for cluster in clusters:
            FILEOUT.write( str(cluster) + "\n\n" )
        FILEOUT.close()

    def show_sample_to_initial_similarity_group_mapping(self):
        self.sample_to_similarity_group_mapping = {sample : [] for sample in self._data_dict}
        for sample in sorted(self._data_dict.keys(), key=lambda x: sample_index(x)):        
            for key in sorted(self.coalesced_band_hash, key=lambda x: band_hash_group_index(x)):            
                if (self.coalesced_band_hash[key] is not None) and (sample in self.coalesced_band_hash[key]):
                    self.sample_to_similarity_group_mapping[sample].append(key)
        print( "\n\nShowing sample to initial similarity group mappings:" )
        for sample in sorted(self.sample_to_similarity_group_mapping.keys(), key=lambda x: sample_index(x)):
            print( "\n %s     =>    %s" % (sample, str(self.sample_to_similarity_group_mapping[sample])) )

    def display_contents_of_all_hash_bins_pre_lsh(self):
        for hplane in self.hash_store:
            print( "\n\n hyperplane: %s" % str(hplane) )
            print( "\n samples in plus bin: %s" % str(self.hash_store[hplane]['plus']) )
            print( "\n samples in minus bin: %s" % str(self.hash_store[hplane]['minus']) )
#-----------------------------  End of Definition for Class LSH --------------------------------


#----------------------  Generate Your Own Data For Experimenting with LSH ------------------------

class DataGenerator(object):
    def __init__(self, *args, **kwargs ):
        if args:
            raise SyntaxError('''DataGenerator can only be called with keyword arguments '''
                              '''for the following keywords: output_csv_file, how_many_similarity_groups '''
                              '''dim, number_of_samples_per_group, and debug''') 
        allowed_keys = 'output_csv_file','dim','covariance','number_of_samples_per_group','how_many_similarity_groups','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise Exception("Wrong keyword used --- check spelling") 
        output_csv_file=dim=covariance=number_of_samples_per_group=debug=None
        if 'output_csv_file' in kwargs :       output_csv_file = kwargs.pop('output_csv_file')
        if 'dim' in kwargs:             dim = kwargs.pop('dim')
        if 'covariance' in kwargs:             covariance = kwargs.pop('covariance')
        if 'debug' in kwargs:                  debug = kwargs.pop('debug')
        if 'how_many_similarity_groups' in kwargs:  
                         how_many_similarity_groups = kwargs.pop('how_many_similarity_groups')
        if 'number_of_samples_per_group' in kwargs:      
                                  number_of_samples_per_group = kwargs.pop('number_of_samples_per_group')
        if output_csv_file:
            self._output_csv_file = output_csv_file
        else:
            raise Exception('''You must specify the name for a csv file for the training data''')
        if dim:
            self.dim = dim
        else:
            raise Exception('''You must specify the dimensionality of your problem''')        
        if covariance is not None: 
            self.covariance = covariance
        else:
            self.covariance = numpy.diag([1] * dim)
        if number_of_samples_per_group:
            self.number_of_samples_per_group = number_of_samples_per_group
        else:
            raise Exception('''You forgot to specify the number of samples per similarity group''')
        if how_many_similarity_groups:
            self.how_many_similarity_groups = how_many_similarity_groups
        else:
            self.how_many_similarity_groups = dim
        if debug:
            self._debug = debug
        else:
            self._debug = 0


    def gen_data_and_write_to_csv(self):
        '''
        Note that a unit cube in N dimensions has 2^N corner points.  The coordinates of all these
        corner points are given by the bit patterns of the integers 0, 1, 2, ...., 2^N - 1.
        For example, in a vector 3-space, a unit cube has 8 corners whose coordinates are given by
        the bit patterns for the integers 0, 1, 2, 3, 4, 5, 6, 7.  These bit patterns would be
        000, 001, 010, 011, 100, 101, 110, 111.

        This script uses only N of the 2^N vertices of a unit cube as the mean vectors for N 
        similarity groups.   These N vertices correspond to the far points on the cube edges that
        emanate at the origin.  For example, when N=3, it uses only 001,010,100 as the three mean
        vectors for the AT MOST 3 similarity groups.  If needed, we can add additional similarity 
        groups by selecting additional coordinate bit patterns from the integers 0 through 2^N - 1.
        '''
        mean_coords = numpy.diag([1] * self.how_many_similarity_groups)
        if self.how_many_similarity_groups < self.dim:
            mean_coords = list(map(lambda x: x + [0] * (self.dim - self.how_many_similarity_groups),
                          [mean_coords[i,].tolist() for i in range(self.how_many_similarity_groups)]))
        else:
            Exception('''The logic for the case when number of similarity groups exceeds '''
                      '''the number of dimensions has not yet been coded''')
        print( "\nShowing the mean vector used for each cluster:" )
        print( str(mean_coords) )
        sample_records = []
        for i in range(self.how_many_similarity_groups):
            k = len(sample_records)
            new_samples = numpy.random.multivariate_normal(mean_coords[i], 
                                                 self.covariance, self.number_of_samples_per_group)
            new_samples = [list(map(float, map(lambda x: "%.3f" % x, sample_coords))) for sample_coords in new_samples]
            for j in range(len(new_samples)):
                sample_records.append('sample' + str(i) + '_' + str(j+k) + ',' 
                                                     + ','.join(list(map(lambda x: str(x), new_samples[j]))) + "\n")
        print("Writing data to the file %s" % self._output_csv_file)
        FILE = open(self._output_csv_file, 'w') 
        list(map(FILE.write, sample_records))
        FILE.close()    
#------------------------  End of Definition for Class DataGenerator ---------------------------


#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':

    '''
    dim = 10
    covar = numpy.diag([0.01] * dim)
    output_file = 'data_for_lsh.csv'
    data_gen = DataGenerator( 
                              output_csv_file   = output_file,
                              how_many_similarity_groups = 10,
                              dim = dim,
                              number_of_samples_per_group = 8,
                              covariance = covar,
                            )

    data_gen.gen_data_and_write_to_csv()
    
    lsh = LocalitySensitiveHashing( datafile = "data_for_lsh.csv",  
                                    dim = 10,
                                    r = 5,                              # number of rows in each band
                                    b = 20,                 # number of bands.   IMPORTANT: Total number of hash fns:  r * b
                                  )
    lsh.get_data_from_csv()
    # print(lsh._data_dict)
    lsh.show_data_for_lsh()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    lsh.display_contents_of_all_hash_bins_pre_lsh()
    lsh.lsh_basic_for_neighborhood_clusters()
    lsh.show_sample_to_similarity_group_mapping()
    '''
    import time
    import generatedDataLoading as loading
    from sklearn.metrics.pairwise import euclidean_distances
    data, groundTruth = loading.get_data("cardio")
    r = 5
    b = 10

    t = time.time()
    lsh = LocalitySensitiveHashing(dim = data.shape[-1], r = r, b = b)
    lsh.get_data_from_nparray(data)
    print("Reading data time: ", time.time() - t)
    t = time.time()
    # lsh.show_data_for_lsh()
    lsh.initialize_hash_store()
    print("Initailizing hashing time: ", time.time() - t)
    t = time.time()
    # print('hashing')


    lsh.hash_all_data_tf(data)
    # lsh.hash_all_data()
    for hplane in lsh.hash_store:
        print(hplane, lsh.hash_store[hplane]['minus'])
    print("Hashing time: ", time.time() - t)
    t = time.time()


    # print('getting neighbors')
    lsh_neighbors = lsh.lsh_basic_for_nearest_neighbors()
    print("Finding hashing neigbors time: ", time.time() - t)
    t = time.time()
    print("lsh_neighbors: ", len(lsh_neighbors))
    anomaly_score = []
    # print('calling auc')
    for i, sample_name in enumerate(lsh_neighbors):
        lsh_index = []
        for item in lsh_neighbors[sample_name]:
            lsh_index.append(int(item))
        band_data = data[lsh_index]
        reasonable_k = min(n_neighbors, len(lsh_index)-1)
        # print('1')
        eu_dis = euclidean_distances(band_data, data[i].reshape(1,-1))
        # print(eu_dis.shape)
        ind = eu_dis.argsort()[reasonable_k]
        one_score = eu_dis[ind]
        anomaly_score.append(one_score)
        print("One score time: ", time.time() - t)
        t = time.time()
        _ = input()
    anomaly_score = np.array(anomaly_score).reshape(-1,1)


