# Results: XLM-RoBERTa
```
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/543d99cd87924ae198ff39c98df31ec1'
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Build finished in 0.20 sec.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Starting training.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Training finished in 3035.44 sec.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:RESULTS:
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:  Validation loss: 0.001263166430112381
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:  Validation accuracy: 0.0
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:  Training loss: 0.001278395001955471
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Transformer initialized with data directory '/data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/8cd952fb97414e9e8ee36bddc35c3325'
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Starting build.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Build finished in 0.23 sec.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Starting prediction.
[2m[36m(pid=1496)[0m INFO:gobbli.experiment.base:Prediction finished in 60.35 sec.
[2m[36m(pid=1496)[0m /usr/local/lib/python3.7/site-packages/ray/pyarrow_files/pyarrow/serialization.py:165: FutureWarning: The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version
[2m[36m(pid=1496)[0m   if isinstance(obj, pd.SparseDataFrame):

```
|    |   valid_loss |   valid_accuracy |   train_loss | multilabel   | labels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | checkpoint                                                                                                                                                     | node_ip_address   | model_params                                                                   |
|---:|-------------:|-----------------:|-------------:|:-------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------|
|  0 |   0.00126317 |                0 |    0.0012784 | True         | ['Absurdism', 'Acid western', 'Action', 'Action Comedy', 'Action Thrillers', 'Action/Adventure', 'Addiction Drama', 'Adult', 'Adventure', 'Adventure Comedy', 'Airplanes and airports', 'Albino bias', 'Alien Film', 'Alien invasion', 'Americana', 'Animal Picture', 'Animals', 'Animated Musical', 'Animated cartoon', 'Animation', 'Anime', 'Anthology', 'Anthropology', 'Anti-war', 'Anti-war film', 'Apocalyptic and post-apocalyptic fiction', 'Archaeology', 'Archives and records', 'Art film', 'Auto racing', 'Avant-garde', 'B-Western', 'B-movie', 'Backstage Musical', 'Baseball', 'Beach Film', 'Beach Party film', 'Bengali Cinema', 'Biker Film', 'Biographical film', 'Biography', 'Biopic [feature]', 'Black comedy', 'Black-and-white', 'Blaxploitation', 'Bloopers & Candid Camera', 'Bollywood', 'Boxing', 'Breakdance', 'British Empire Film', 'British New Wave', 'Bruceploitation', 'Buddy cop', 'Buddy film', 'Business', 'C-Movie', 'Camp', 'Caper story', 'Cavalry Film', 'Chase Movie', 'Childhood Drama', "Children's", "Children's Entertainment", "Children's Fantasy", "Children's Issues", "Children's/Family", 'Chinese Movies', 'Christian film', 'Christmas movie', 'Clay animation', 'Cold War', 'Combat Films', 'Comdedy', 'Comedy', 'Comedy Thriller', 'Comedy Western', 'Comedy film', 'Comedy horror', 'Comedy of Errors', 'Comedy of manners', 'Comedy-drama', 'Coming of age', 'Coming-of-age film', 'Computer Animation', 'Computers', 'Concert film', 'Conspiracy fiction', 'Costume Adventure', 'Costume Horror', 'Costume drama', 'Courtroom Comedy', 'Courtroom Drama', 'Creature Film', 'Crime', 'Crime Comedy', 'Crime Drama', 'Crime Fiction', 'Crime Thriller', 'Cult', 'Culture & Society', 'Cyberpunk', 'Czechoslovak New Wave', 'Dance', 'Demonic child', 'Detective', 'Detective fiction', 'Disaster', 'Docudrama', 'Documentary', 'Dogme 95', 'Domestic Comedy', 'Doomsday film', 'Drama', 'Dystopia', 'Ealing Comedies', 'Early Black Cinema', 'Education', 'Educational', 'Ensemble Film', 'Environmental Science', 'Epic', 'Epic Western', 'Erotic Drama', 'Erotic thriller', 'Erotica', 'Escape Film', 'Essay Film', 'Existentialism', 'Experimental film', 'Exploitation', 'Expressionism', 'Extreme Sports', 'Fairy tale', 'Family & Personal Relationships', 'Family Drama', 'Family Film', 'Family-Oriented Adventure', 'Fan film', 'Fantasy', 'Fantasy Adventure', 'Fantasy Comedy', 'Fantasy Drama', 'Feature film', 'Female buddy film', 'Feminist Film', 'Fictional film', 'Filipino', 'Filipino Movies', 'Film', 'Film & Television History', 'Film adaptation', 'Film noir', 'Film à clef', 'Film-Opera', 'Filmed Play', 'Finance & Investing', 'Foreign legion', 'Future noir', 'Gangster Film', 'Gay', 'Gay Interest', 'Gay Themed', 'Gay pornography', 'Gender Issues', 'Giallo', 'Glamorized Spy Film', 'Goat gland', 'Gothic Film', 'Graphic & Applied Arts', 'Gross out', 'Gross-out film', 'Gulf War', 'Hagiography', 'Hardcore pornography', 'Haunted House Film', 'Health & Fitness', 'Heaven-Can-Wait Fantasies', 'Heavenly Comedy', 'Heist', 'Hip hop movies', 'Historical Documentaries', 'Historical Epic', 'Historical drama', 'Historical fiction', 'History', 'Holiday Film', 'Homoeroticism', 'Horror', 'Horror Comedy', 'Horse racing', 'Humour', 'Hybrid Western', 'Illnesses & Disabilities', 'Indian Western', 'Indie', 'Inspirational Drama', 'Instrumental Music', 'Interpersonal Relationships', 'Inventions & Innovations', 'Japanese Movies', 'Journalism', 'Jukebox musical', 'Jungle Film', 'Juvenile Delinquency Film', 'Kafkaesque', 'Kitchen sink realism', 'LGBT', 'Language & Literature', 'Law & Crime', 'Legal drama', 'Libraries and librarians', 'Linguistics', 'Live action', 'Malayalam Cinema', 'Marriage Drama', 'Martial Arts Film', 'Master Criminal Films', 'Media Satire', 'Media Studies', 'Medical fiction', 'Melodrama', 'Mockumentary', 'Mondo film', 'Monster', 'Monster movie', 'Movie serial', 'Movies About Gladiators', 'Mumblecore', 'Music', 'Musical', 'Musical Drama', 'Musical comedy', 'Mystery', 'Mythological Fantasy', 'Natural disaster', 'Natural horror films', 'Nature', 'Neo-noir', 'Neorealism', 'New Hollywood', 'New Queer Cinema', 'News', 'Ninja movie', 'Northern', 'Nuclear warfare', 'Operetta', 'Outlaw', 'Outlaw biker film', 'Parkour in popular culture', 'Parody', 'Patriotic film', 'Period Horror', 'Period piece', 'Pinku eiga', 'Plague', 'Point of view shot', 'Political cinema', 'Political drama', 'Political satire', 'Political thriller', 'Pornographic movie', 'Pornography', 'Pre-Code', 'Prison', 'Prison escape', 'Prison film', 'Private military company', 'Propaganda film', 'Psycho-biddy', 'Psychological horror', 'Psychological thriller', 'Punk rock', 'Race movie', 'Reboot', 'Religious Film', 'Remake', 'Revenge', 'Revisionist Fairy Tale', 'Revisionist Western', 'Road movie', 'Road-Horror', 'Roadshow theatrical release', 'Roadshow/Carny', 'Rockumentary', 'Romance Film', 'Romantic comedy', 'Romantic drama', 'Romantic fantasy', 'Samurai cinema', 'Satire', 'School story', 'Sci Fi Pictures original films', 'Sci-Fi Adventure', 'Sci-Fi Horror', 'Sci-Fi Thriller', 'Science Fiction', 'Science fiction Western', 'Screwball comedy', 'Sex comedy', 'Sexploitation', 'Short Film', 'Silent film', 'Silhouette animation', 'Singing cowboy', 'Slapstick', 'Slasher', 'Slice of life story', 'Social issues', 'Social problem film', 'Softcore Porn', 'Space opera', 'Space western', 'Spaghetti Western', 'Splatter film', 'Sponsored film', 'Sports', 'Spy', 'Stand-up comedy', 'Star vehicle', 'Statutory rape', 'Steampunk', 'Stoner film', 'Stop motion', 'Superhero', 'Superhero movie', 'Supermarionation', 'Supernatural', 'Surrealism', 'Suspense', 'Swashbuckler films', 'Sword and Sandal', 'Sword and sorcery', 'Sword and sorcery films', 'Tamil cinema', 'Teen', 'Television movie', 'The Netherlands in World War II', 'Therimin music', 'Thriller', 'Time travel', 'Tokusatsu', 'Tollywood', 'Tragedy', 'Tragicomedy', 'Travel', 'Vampire movies', 'War film', 'Werewolf fiction', 'Western', 'Whodunit', 'Women in prison films', 'Workplace Comedy', 'World History', 'World cinema', 'Wuxia', 'Z movie', 'Zombie Film'] | /data/users/jnance/gobbli/benchmark/benchmark_data/model/Transformer/543d99cd87924ae198ff39c98df31ec1/train/ddada31d783c4802a9b8fc9e438fbb92/output/checkpoint | 172.80.10.2       | {'transformer_model': 'XLMRoberta', 'transformer_weights': 'xlm-roberta-base'} |
```
Metrics:
--------
Weighted F1 Score: 0.0
Weighted Precision Score: 0.0
Weighted Recall Score: 0.0
Accuracy: 0.026537140149271412

Classification Report:
----------------------
                                          precision    recall  f1-score   support

                               Absurdism       0.00      0.00      0.00         1
                            Acid western       0.00      0.00      0.00         0
                                  Action       0.00      0.00      0.00      1051
                           Action Comedy       0.00      0.00      0.00         3
                        Action Thrillers       0.00      0.00      0.00        11
                        Action/Adventure       0.00      0.00      0.00       263
                         Addiction Drama       0.00      0.00      0.00         2
                                   Adult       0.00      0.00      0.00        23
                               Adventure       0.00      0.00      0.00       411
                        Adventure Comedy       0.00      0.00      0.00         3
                  Airplanes and airports       0.00      0.00      0.00         0
                             Albino bias       0.00      0.00      0.00         0
                              Alien Film       0.00      0.00      0.00         4
                          Alien invasion       0.00      0.00      0.00         0
                               Americana       0.00      0.00      0.00         1
                          Animal Picture       0.00      0.00      0.00         4
                                 Animals       0.00      0.00      0.00         2
                        Animated Musical       0.00      0.00      0.00         0
                        Animated cartoon       0.00      0.00      0.00         4
                               Animation       0.00      0.00      0.00       442
                                   Anime       0.00      0.00      0.00        32
                               Anthology       0.00      0.00      0.00         0
                            Anthropology       0.00      0.00      0.00         0
                                Anti-war       0.00      0.00      0.00         0
                           Anti-war film       0.00      0.00      0.00         0
Apocalyptic and post-apocalyptic fiction       0.00      0.00      0.00         0
                             Archaeology       0.00      0.00      0.00         0
                    Archives and records       0.00      0.00      0.00         0
                                Art film       0.00      0.00      0.00         8
                             Auto racing       0.00      0.00      0.00         3
                             Avant-garde       0.00      0.00      0.00         4
                               B-Western       0.00      0.00      0.00         1
                                 B-movie       0.00      0.00      0.00        37
                       Backstage Musical       0.00      0.00      0.00         0
                                Baseball       0.00      0.00      0.00         0
                              Beach Film       0.00      0.00      0.00         0
                        Beach Party film       0.00      0.00      0.00         0
                          Bengali Cinema       0.00      0.00      0.00         0
                              Biker Film       0.00      0.00      0.00         0
                       Biographical film       0.00      0.00      0.00       109
                               Biography       0.00      0.00      0.00        41
                        Biopic [feature]       0.00      0.00      0.00        16
                            Black comedy       0.00      0.00      0.00        44
                         Black-and-white       0.00      0.00      0.00        12
                          Blaxploitation       0.00      0.00      0.00         8
                Bloopers & Candid Camera       0.00      0.00      0.00         0
                               Bollywood       0.00      0.00      0.00        92
                                  Boxing       0.00      0.00      0.00         3
                              Breakdance       0.00      0.00      0.00         0
                     British Empire Film       0.00      0.00      0.00         0
                        British New Wave       0.00      0.00      0.00         0
                         Bruceploitation       0.00      0.00      0.00         0
                               Buddy cop       0.00      0.00      0.00         0
                              Buddy film       0.00      0.00      0.00        11
                                Business       0.00      0.00      0.00         0
                                 C-Movie       0.00      0.00      0.00         0
                                    Camp       0.00      0.00      0.00         0
                             Caper story       0.00      0.00      0.00         1
                            Cavalry Film       0.00      0.00      0.00         0
                             Chase Movie       0.00      0.00      0.00         1
                         Childhood Drama       0.00      0.00      0.00         4
                              Children's       0.00      0.00      0.00        19
                Children's Entertainment       0.00      0.00      0.00         0
                      Children's Fantasy       0.00      0.00      0.00         5
                       Children's Issues       0.00      0.00      0.00         0
                       Children's/Family       0.00      0.00      0.00        17
                          Chinese Movies       0.00      0.00      0.00       104
                          Christian film       0.00      0.00      0.00         1
                         Christmas movie       0.00      0.00      0.00         2
                          Clay animation       0.00      0.00      0.00         1
                                Cold War       0.00      0.00      0.00         0
                            Combat Films       0.00      0.00      0.00         4
                                 Comdedy       0.00      0.00      0.00         0
                                  Comedy       0.00      0.00      0.00       856
                         Comedy Thriller       0.00      0.00      0.00         4
                          Comedy Western       0.00      0.00      0.00         2
                             Comedy film       0.00      0.00      0.00      1107
                           Comedy horror       0.00      0.00      0.00         2
                        Comedy of Errors       0.00      0.00      0.00         3
                       Comedy of manners       0.00      0.00      0.00         5
                            Comedy-drama       0.00      0.00      0.00        69
                           Coming of age       0.00      0.00      0.00        50
                      Coming-of-age film       0.00      0.00      0.00         2
                      Computer Animation       0.00      0.00      0.00        19
                               Computers       0.00      0.00      0.00         0
                            Concert film       0.00      0.00      0.00         2
                      Conspiracy fiction       0.00      0.00      0.00         0
                       Costume Adventure       0.00      0.00      0.00         0
                          Costume Horror       0.00      0.00      0.00         0
                           Costume drama       0.00      0.00      0.00         3
                        Courtroom Comedy       0.00      0.00      0.00         0
                         Courtroom Drama       0.00      0.00      0.00         6
                           Creature Film       0.00      0.00      0.00         5
                                   Crime       0.00      0.00      0.00         7
                            Crime Comedy       0.00      0.00      0.00         5
                             Crime Drama       0.00      0.00      0.00        21
                           Crime Fiction       0.00      0.00      0.00       683
                          Crime Thriller       0.00      0.00      0.00       121
                                    Cult       0.00      0.00      0.00        25
                       Culture & Society       0.00      0.00      0.00        17
                               Cyberpunk       0.00      0.00      0.00         0
                   Czechoslovak New Wave       0.00      0.00      0.00         0
                                   Dance       0.00      0.00      0.00         9
                           Demonic child       0.00      0.00      0.00         0
                               Detective       0.00      0.00      0.00        19
                       Detective fiction       0.00      0.00      0.00        19
                                Disaster       0.00      0.00      0.00        23
                               Docudrama       0.00      0.00      0.00        12
                             Documentary       0.00      0.00      0.00       455
                                Dogme 95       0.00      0.00      0.00         2
                         Domestic Comedy       0.00      0.00      0.00         6
                           Doomsday film       0.00      0.00      0.00         1
                                   Drama       0.00      0.00      0.00      3499
                                Dystopia       0.00      0.00      0.00         0
                         Ealing Comedies       0.00      0.00      0.00         0
                      Early Black Cinema       0.00      0.00      0.00         0
                               Education       0.00      0.00      0.00         0
                             Educational       0.00      0.00      0.00         3
                           Ensemble Film       0.00      0.00      0.00         7
                   Environmental Science       0.00      0.00      0.00         1
                                    Epic       0.00      0.00      0.00        17
                            Epic Western       0.00      0.00      0.00         0
                            Erotic Drama       0.00      0.00      0.00         2
                         Erotic thriller       0.00      0.00      0.00        19
                                 Erotica       0.00      0.00      0.00        14
                             Escape Film       0.00      0.00      0.00         1
                              Essay Film       0.00      0.00      0.00         0
                          Existentialism       0.00      0.00      0.00         0
                       Experimental film       0.00      0.00      0.00         9
                            Exploitation       0.00      0.00      0.00         2
                           Expressionism       0.00      0.00      0.00         0
                          Extreme Sports       0.00      0.00      0.00         1
                              Fairy tale       0.00      0.00      0.00         3
         Family & Personal Relationships       0.00      0.00      0.00         1
                            Family Drama       0.00      0.00      0.00        44
                             Family Film       0.00      0.00      0.00       462
               Family-Oriented Adventure       0.00      0.00      0.00         2
                                Fan film       0.00      0.00      0.00         5
                                 Fantasy       0.00      0.00      0.00       251
                       Fantasy Adventure       0.00      0.00      0.00         1
                          Fantasy Comedy       0.00      0.00      0.00         4
                           Fantasy Drama       0.00      0.00      0.00         1
                            Feature film       0.00      0.00      0.00         0
                       Female buddy film       0.00      0.00      0.00         0
                           Feminist Film       0.00      0.00      0.00         3
                          Fictional film       0.00      0.00      0.00         4
                                Filipino       0.00      0.00      0.00         1
                         Filipino Movies       0.00      0.00      0.00        33
                                    Film       0.00      0.00      0.00         1
               Film & Television History       0.00      0.00      0.00         1
                         Film adaptation       0.00      0.00      0.00        60
                               Film noir       0.00      0.00      0.00        37
                             Film à clef       0.00      0.00      0.00         0
                              Film-Opera       0.00      0.00      0.00         0
                             Filmed Play       0.00      0.00      0.00         0
                     Finance & Investing       0.00      0.00      0.00         0
                          Foreign legion       0.00      0.00      0.00         0
                             Future noir       0.00      0.00      0.00         0
                           Gangster Film       0.00      0.00      0.00        16
                                     Gay       0.00      0.00      0.00         7
                            Gay Interest       0.00      0.00      0.00         7
                              Gay Themed       0.00      0.00      0.00        13
                         Gay pornography       0.00      0.00      0.00         4
                           Gender Issues       0.00      0.00      0.00         1
                                  Giallo       0.00      0.00      0.00         2
                     Glamorized Spy Film       0.00      0.00      0.00         3
                              Goat gland       0.00      0.00      0.00         0
                             Gothic Film       0.00      0.00      0.00         3
                  Graphic & Applied Arts       0.00      0.00      0.00         0
                               Gross out       0.00      0.00      0.00         0
                          Gross-out film       0.00      0.00      0.00         0
                                Gulf War       0.00      0.00      0.00         0
                             Hagiography       0.00      0.00      0.00         2
                    Hardcore pornography       0.00      0.00      0.00         0
                      Haunted House Film       0.00      0.00      0.00         4
                        Health & Fitness       0.00      0.00      0.00         0
               Heaven-Can-Wait Fantasies       0.00      0.00      0.00         1
                         Heavenly Comedy       0.00      0.00      0.00         0
                                   Heist       0.00      0.00      0.00         6
                          Hip hop movies       0.00      0.00      0.00         4
                Historical Documentaries       0.00      0.00      0.00         1
                         Historical Epic       0.00      0.00      0.00         2
                        Historical drama       0.00      0.00      0.00        34
                      Historical fiction       0.00      0.00      0.00        67
                                 History       0.00      0.00      0.00        52
                            Holiday Film       0.00      0.00      0.00         3
                           Homoeroticism       0.00      0.00      0.00         0
                                  Horror       0.00      0.00      0.00       640
                           Horror Comedy       0.00      0.00      0.00         8
                            Horse racing       0.00      0.00      0.00         1
                                  Humour       0.00      0.00      0.00         0
                          Hybrid Western       0.00      0.00      0.00         0
                Illnesses & Disabilities       0.00      0.00      0.00         1
                          Indian Western       0.00      0.00      0.00         0
                                   Indie       0.00      0.00      0.00       208
                     Inspirational Drama       0.00      0.00      0.00         1
                      Instrumental Music       0.00      0.00      0.00         0
             Interpersonal Relationships       0.00      0.00      0.00         1
                Inventions & Innovations       0.00      0.00      0.00         0
                         Japanese Movies       0.00      0.00      0.00       136
                              Journalism       0.00      0.00      0.00         0
                         Jukebox musical       0.00      0.00      0.00         0
                             Jungle Film       0.00      0.00      0.00         2
               Juvenile Delinquency Film       0.00      0.00      0.00         3
                              Kafkaesque       0.00      0.00      0.00         0
                    Kitchen sink realism       0.00      0.00      0.00         0
                                    LGBT       0.00      0.00      0.00        69
                   Language & Literature       0.00      0.00      0.00         0
                             Law & Crime       0.00      0.00      0.00         2
                             Legal drama       0.00      0.00      0.00         0
                Libraries and librarians       0.00      0.00      0.00         0
                             Linguistics       0.00      0.00      0.00         0
                             Live action       0.00      0.00      0.00         0
                        Malayalam Cinema       0.00      0.00      0.00         0
                          Marriage Drama       0.00      0.00      0.00         1
                       Martial Arts Film       0.00      0.00      0.00        52
                   Master Criminal Films       0.00      0.00      0.00         0
                            Media Satire       0.00      0.00      0.00         1
                           Media Studies       0.00      0.00      0.00         0
                         Medical fiction       0.00      0.00      0.00         2
                               Melodrama       0.00      0.00      0.00        27
                            Mockumentary       0.00      0.00      0.00         8
                              Mondo film       0.00      0.00      0.00         1
                                 Monster       0.00      0.00      0.00         0
                           Monster movie       0.00      0.00      0.00        10
                            Movie serial       0.00      0.00      0.00         0
                 Movies About Gladiators       0.00      0.00      0.00         0
                              Mumblecore       0.00      0.00      0.00         2
                                   Music       0.00      0.00      0.00        68
                                 Musical       0.00      0.00      0.00       312
                           Musical Drama       0.00      0.00      0.00         5
                          Musical comedy       0.00      0.00      0.00         3
                                 Mystery       0.00      0.00      0.00       308
                    Mythological Fantasy       0.00      0.00      0.00         2
                        Natural disaster       0.00      0.00      0.00         0
                    Natural horror films       0.00      0.00      0.00         5
                                  Nature       0.00      0.00      0.00         4
                                Neo-noir       0.00      0.00      0.00         1
                              Neorealism       0.00      0.00      0.00         0
                           New Hollywood       0.00      0.00      0.00         0
                        New Queer Cinema       0.00      0.00      0.00         0
                                    News       0.00      0.00      0.00         4
                             Ninja movie       0.00      0.00      0.00         0
                                Northern       0.00      0.00      0.00         0
                         Nuclear warfare       0.00      0.00      0.00         0
                                Operetta       0.00      0.00      0.00         1
                                  Outlaw       0.00      0.00      0.00         0
                       Outlaw biker film       0.00      0.00      0.00         0
              Parkour in popular culture       0.00      0.00      0.00         0
                                  Parody       0.00      0.00      0.00        25
                          Patriotic film       0.00      0.00      0.00         0
                           Period Horror       0.00      0.00      0.00         0
                            Period piece       0.00      0.00      0.00        47
                              Pinku eiga       0.00      0.00      0.00         6
                                  Plague       0.00      0.00      0.00         0
                      Point of view shot       0.00      0.00      0.00         0
                        Political cinema       0.00      0.00      0.00        19
                         Political drama       0.00      0.00      0.00        36
                        Political satire       0.00      0.00      0.00         6
                      Political thriller       0.00      0.00      0.00        10
                      Pornographic movie       0.00      0.00      0.00        19
                             Pornography       0.00      0.00      0.00         1
                                Pre-Code       0.00      0.00      0.00         5
                                  Prison       0.00      0.00      0.00         2
                           Prison escape       0.00      0.00      0.00         0
                             Prison film       0.00      0.00      0.00         1
                Private military company       0.00      0.00      0.00         0
                         Propaganda film       0.00      0.00      0.00         7
                            Psycho-biddy       0.00      0.00      0.00         0
                    Psychological horror       0.00      0.00      0.00         0
                  Psychological thriller       0.00      0.00      0.00        73
                               Punk rock       0.00      0.00      0.00         1
                              Race movie       0.00      0.00      0.00         0
                                  Reboot       0.00      0.00      0.00         1
                          Religious Film       0.00      0.00      0.00         7
                                  Remake       0.00      0.00      0.00        10
                                 Revenge       0.00      0.00      0.00         1
                  Revisionist Fairy Tale       0.00      0.00      0.00         0
                     Revisionist Western       0.00      0.00      0.00         1
                              Road movie       0.00      0.00      0.00        16
                             Road-Horror       0.00      0.00      0.00         0
             Roadshow theatrical release       0.00      0.00      0.00         0
                          Roadshow/Carny       0.00      0.00      0.00         0
                            Rockumentary       0.00      0.00      0.00         6
                            Romance Film       0.00      0.00      0.00      1022
                         Romantic comedy       0.00      0.00      0.00       184
                          Romantic drama       0.00      0.00      0.00       187
                        Romantic fantasy       0.00      0.00      0.00         1
                          Samurai cinema       0.00      0.00      0.00         2
                                  Satire       0.00      0.00      0.00        21
                            School story       0.00      0.00      0.00         0
          Sci Fi Pictures original films       0.00      0.00      0.00         2
                        Sci-Fi Adventure       0.00      0.00      0.00         3
                           Sci-Fi Horror       0.00      0.00      0.00         5
                         Sci-Fi Thriller       0.00      0.00      0.00         3
                         Science Fiction       0.00      0.00      0.00       255
                 Science fiction Western       0.00      0.00      0.00         0
                        Screwball comedy       0.00      0.00      0.00         7
                              Sex comedy       0.00      0.00      0.00         7
                           Sexploitation       0.00      0.00      0.00         5
                              Short Film       0.00      0.00      0.00       751
                             Silent film       0.00      0.00      0.00       306
                    Silhouette animation       0.00      0.00      0.00         0
                          Singing cowboy       0.00      0.00      0.00         0
                               Slapstick       0.00      0.00      0.00        12
                                 Slasher       0.00      0.00      0.00        55
                     Slice of life story       0.00      0.00      0.00         5
                           Social issues       0.00      0.00      0.00         7
                     Social problem film       0.00      0.00      0.00         5
                           Softcore Porn       0.00      0.00      0.00         2
                             Space opera       0.00      0.00      0.00         0
                           Space western       0.00      0.00      0.00         0
                       Spaghetti Western       0.00      0.00      0.00         7
                           Splatter film       0.00      0.00      0.00         1
                          Sponsored film       0.00      0.00      0.00         3
                                  Sports       0.00      0.00      0.00        93
                                     Spy       0.00      0.00      0.00        27
                         Stand-up comedy       0.00      0.00      0.00         0
                            Star vehicle       0.00      0.00      0.00         0
                          Statutory rape       0.00      0.00      0.00         0
                               Steampunk       0.00      0.00      0.00         0
                             Stoner film       0.00      0.00      0.00         1
                             Stop motion       0.00      0.00      0.00         3
                               Superhero       0.00      0.00      0.00         4
                         Superhero movie       0.00      0.00      0.00         7
                        Supermarionation       0.00      0.00      0.00         0
                            Supernatural       0.00      0.00      0.00        36
                              Surrealism       0.00      0.00      0.00         1
                                Suspense       0.00      0.00      0.00        50
                      Swashbuckler films       0.00      0.00      0.00         0
                        Sword and Sandal       0.00      0.00      0.00         0
                       Sword and sorcery       0.00      0.00      0.00         1
                 Sword and sorcery films       0.00      0.00      0.00         0
                            Tamil cinema       0.00      0.00      0.00         1
                                    Teen       0.00      0.00      0.00        36
                        Television movie       0.00      0.00      0.00        63
         The Netherlands in World War II       0.00      0.00      0.00         0
                          Therimin music       0.00      0.00      0.00         0
                                Thriller       0.00      0.00      0.00      1036
                             Time travel       0.00      0.00      0.00         2
                               Tokusatsu       0.00      0.00      0.00         0
                               Tollywood       0.00      0.00      0.00         2
                                 Tragedy       0.00      0.00      0.00         4
                             Tragicomedy       0.00      0.00      0.00         3
                                  Travel       0.00      0.00      0.00         4
                          Vampire movies       0.00      0.00      0.00         1
                                War film       0.00      0.00      0.00       194
                        Werewolf fiction       0.00      0.00      0.00         0
                                 Western       0.00      0.00      0.00       153
                                Whodunit       0.00      0.00      0.00         2
                   Women in prison films       0.00      0.00      0.00         1
                        Workplace Comedy       0.00      0.00      0.00         3
                           World History       0.00      0.00      0.00         0
                            World cinema       0.00      0.00      0.00       477
                                   Wuxia       0.00      0.00      0.00        12
                                 Z movie       0.00      0.00      0.00         0
                             Zombie Film       0.00      0.00      0.00        11

                               micro avg       0.00      0.00      0.00     18361
                               macro avg       0.00      0.00      0.00     18361
                            weighted avg       0.00      0.00      0.00     18361
                             samples avg       0.00      0.00      0.00     18361


```

![Results](XLM-RoBERTa/plot.png)
---