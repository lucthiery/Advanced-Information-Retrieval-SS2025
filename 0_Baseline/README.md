# CLEF2025--CheckThat-Lab
This repository contains the _dataset_, _format checker, scorer and baselines_ for each task of the [CLEF2025-CheckThat! Lab](https://checkthat.gitlab.io/).

The **2025 edition** of the lab focuses on tasks that are the core of the verification pipeline again as well as auxiliary tasks:
Task 1 is on identification of subjectivity (a follow up of the \ct{} 2024 edition),
Task 2 is on claim normalization,
Task 3 addresses fact-checking numerical claims, and
Task 4 focuses on scientific web discourse processing.
  - [Task 1 Subjectivity in news articles](task1)
      - **Arabic**
      - **Bulgarian**
      - **English**
      - **German**
      - **Italian**
  - [Task 2 Claim Normalization](task2)
    - The task is offered in 20 languages: English, Arabic, Bengali, Czech, German, Greek, French, Hindi, Korean, Marathi, Indonesian, Dutch, Punjabi, Polish, Portuguese, Romanian, Spanish, Tamil, Telugu, Thai.
  - [Task 3: Fact-Checking Numerical Claims:](task3)
    - **Arabic**
    - **English**
    - **Spanish**
  - [Task 4 Scientific Web Discourse Processing](task4)
      - **English**

# Licensing

Please check the task-specific directory for the licensing of the respective dataset.

# Credits

**Lab Organizers:** Please find on the task website: https://checkthat.gitlab.io/

# Contact
Please join on Slack channel for any queries: [https://join.slack.com/t/ct-participants/shared_invite/zt-1me6ywqxu-0Q~r7vCm6gfmrmN9xuOO7A](https://join.slack.com/t/ct-participants/shared_invite/zt-1me6ywqxu-0Q~r7vCm6gfmrmN9xuOO7A)


Or send an email to: *clef-factcheck@googlegroups.com*


# Citation
TBA



## Additional Resources (Tools/Source code)
We listed the following tools/source code, which might be helpful to run the experiments.
* https://huggingface.co/docs/transformers/training
* https://github.com/Tiiiger/bert_score
* https://github.com/utahnlp/x-fact
* https://github.com/firojalam/COVID-19-disinformation/tree/master/bin


## Recommended reading
The following papers might be useful. We have not provided exhaustive list. But these could be a good start.<br>
[Download bibliography](bibtex/bibliography.bib)

**Multimodal**
* Gullal Singh Cheema, Sherzod Hakimov, Abdul Sittar, Eric Müller-Budack, Christian Otto, and Ralph Ewerth. 2022. **[MM-Claims: A Dataset for Multimodal Claim Detection in Social Media](https://aclanthology.org/2022.findings-naacl.72/)**. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 962–979, Seattle, United States. Association for Computational Linguistics.
* Firoj Alam, Stefano Cresci, Tanmoy Chakraborty, Fabrizio Silvestri, Dimiter Dimitrov, Giovanni Da San Martino, Shaden Shaar, Hamed Firooz, and Preslav Nakov. 2022. **[A Survey on Multimodal Disinformation Detection](https://aclanthology.org/2022.coling-1.576/)**. In Proceedings of the 29th International Conference on Computational Linguistics, pages 6625–6643, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.
* Gullal Singh Cheema, Sherzod Hakimov, Eric Müller-Budack, and Ralph Ewerth. 2021. **[On the Role of Images for Analyzing Claims in Social Media](https://ceur-ws.org/Vol-2829/paper3.pdf)**. In Proceedings of the 2nd International Workshop on Cross-lingual Event-centric Open Analytics co-located with the 30th The Web Conference (WWW 2021).
* Dimitrina Zlatkova, Preslav Nakov, and Ivan Koychev. 2019. **[Fact-Checking Meets Fauxtography: Verifying Claims About Images](https://aclanthology.org/D19-1216/)**. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2099–2108, Hong Kong, China. Association for Computational Linguistics.

**Fact-checking**
* Nakov, Preslav, David Corney, Maram Hasanain, Firoj Alam, and Tamer Elsayed. **["Automated Fact-Checking for Assisting Human Fact-Checkers."](https://www.ijcai.org/proceedings/2021/0619.pdf)** in IJCAI, 2021.
* Shaar, Shaden, Firoj Alam, Giovanni Da San Martino, and Preslav Nakov. **"Assisting the Human Fact-Checkers: Detecting All Previously Fact-Checked Claims in a Document."** arXiv preprint arXiv:2109.07410 (2021).
* Shaar, Shaden, Firoj Alam, Giovanni Da San Martino, and Preslav Nakov. **"The role of context in detecting previously fact-checked claims."** arXiv preprint arXiv:2104.07423 (2021).
* Alam, Firoj, Shaden Shaar, Fahim Dalvi, Hassan Sajjad, Alex Nikolov, Hamdy Mubarak, Giovanni Da San Martino et al. **["Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society."](https://aclanthology.org/2021.findings-emnlp.56.pdf)** In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 611-649. 2021.
* Shaar, Shaden, Firoj Alam, Giovanni Da San Martino, Alex Nikolov, Wajdi Zaghouani, Preslav Nakov, and Anna Feldman. **"Findings of the NLP4IF-2021 Shared Tasks on Fighting the COVID-19 Infodemic and Censorship Detection."** In Proceedings of the Fourth Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda, pp. 82-92. 2021.
* Nakov, Preslav, Firoj Alam, Shaden Shaar, Giovanni Da San Martino, and Yifan Zhang. **"A Second Pandemic? Analysis of Fake News about COVID-19 Vaccines in Qatar."** In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pp. 1010-1021. 2021.
* Nakov, Preslav, Firoj Alam, Shaden Shaar, Giovanni Da San Martino, and Yifan Zhang. **"COVID-19 in Bulgarian social media: Factuality, harmfulness, propaganda, and framing."** In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pp. 997-1009. 2021.

**Tasks papers from previous years**
* Nakov, Preslav, Giovanni Da San Martino, Tamer Elsayed, Alberto Barrón-Cedeño, Rubén Míguez, Shaden Shaar, Firoj Alam et al. **"Overview of the CLEF–2021 CheckThat! Lab on Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 264-291. Springer, Cham, 2021.
* Shaar, Shaden, Maram Hasanain, Bayan Hamdan, Zien Sheikh Ali, Fatima Haouari, Alex Nikolov, Mücahid Kutlu et al. **"Overview of the CLEF-2021 CheckThat! lab task 1 on check-worthiness estimation in tweets and political debates."** In CLEF (Working Notes). 2021.
* Shahi, Gautam Kishore, Julia Maria Struß, and Thomas Mandl. **"Overview of the CLEF-2021 CheckThat! lab task 3 on fake news detection."** Working Notes of CLEF (2021).
* Shaar, Shaden, Fatima Haouari, Watheq Mansour, Maram Hasanain, Nikolay Babulkov, Firoj Alam, Giovanni Da San Martino, Tamer Elsayed, and Preslav Nakov. **"Overview of the CLEF-2021 CheckThat! lab task 2 on detecting previously fact-checked claims in tweets and political debates."** In CLEF (Working Notes). 2021.
* Barrón-Cedeño, Alberto, Tamer Elsayed, Preslav Nakov, Giovanni Da San Martino, Maram Hasanain, Reem Suwaileh, Fatima Haouari et al. **"Overview of CheckThat! 2020: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 215-236. Springer, Cham, 2020.
* Shaar, Shaden, Alex Nikolov, Nikolay Babulkov, Firoj Alam, Alberto Barrón-Cedeno, Tamer Elsayed, Maram Hasanain et al. **"Overview of CheckThat! 2020 English: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages. 2020.
* Hasanain, Maram, Fatima Haouari, Reem Suwaileh, Zien Sheikh Ali, Bayan Hamdan, Tamer Elsayed, Alberto Barrón-Cedeno, Giovanni Da San Martino, and Preslav Nakov. **"Overview of CheckThat! 2020 Arabic: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages. 2020.
* Elsayed, Tamer, Preslav Nakov, Alberto Barrón-Cedeno, Maram Hasanain, Reem Suwaileh, Giovanni Da San Martino, and Pepa Atanasova. **"Overview of the CLEF-2019 CheckThat! Lab: automatic identification and verification of claims."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 301-321. Springer, Cham, 2019.
* Elsayed, Tamer, Preslav Nakov, Alberto Barrón-Cedeno, Maram Hasanain, Reem Suwaileh, Giovanni Da San Martino, and Pepa Atanasova. **"CheckThat! at CLEF 2019: Automatic identification and verification of claims."** In European Conference on Information Retrieval, pp. 309-315. Springer, Cham, 2019.
* Atanasova, Pepa, Preslav Nakov, Georgi Karadzhov, Mitra Mohtarami, and Giovanni Da San Martino. **"Overview of the CLEF-2019 CheckThat! Lab: Automatic Identification and Verification of Claims. Task 1: Check-Worthiness."** CLEF (Working Notes) 2380 (2019).
* Nakov, Preslav, Alberto Barrón-Cedeno, Tamer Elsayed, Reem Suwaileh, Lluís Màrquez, Wajdi Zaghouani, Pepa Atanasova, Spas Kyuchukov, and Giovanni Da San Martino. **"Overview of the CLEF-2018 CheckThat! Lab on automatic identification and verification of political claims."** In International conference of the cross-language evaluation forum for european languages, pp. 372-387. Springer, Cham, 2018.
* Barrón-Cedeno, Alberto, Tamer Elsayed, Reem Suwaileh, Lluís Màrquez, Pepa Atanasova, Wajdi Zaghouani, Spas Kyuchukov, Giovanni Da San Martino, and Preslav Nakov. **"Overview of the CLEF-2018 CheckThat! Lab on Automatic Identification and Verification of Political Claims. Task 2: Factuality."**   CLEF (Working Notes) 2125 (2018).
* Atanasova, Pepa, Alberto Barron-Cedeno, Tamer Elsayed, Reem Suwaileh, Wajdi Zaghouani, Spas Kyuchukov, Giovanni Da San Martino, and Preslav Nakov. **"Overview of the CLEF-2018 CheckThat! lab on automatic identification and verification of political claims. Task 1: Check-worthiness."** arXiv preprint arXiv:1808.05542 (2018).
