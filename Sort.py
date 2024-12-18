from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
syllabus_points = {
    """ 
        animals - external and internal fertilisation, plants - asexual and sexual reproduction
        fungi -  budding, spores, bacteria - binary fission, protists - binary fission, budding
        fertilisation, implantation and hormonal control of pregnancy and birth in
        mammals, scientific knowledge on plant and animal reproduction in agriculture 
    """: "MOD 5 IQ 1",
    """
        cell replication, mitosis and meiosis, DNA replication with Watson and Crick DNA model, including nucleotide composition, pairing and
        bonding, cell replication processes on the continuity of species
    """: "MOD 5 IQ 2",
    """
        DNA exists in eukaryotes and prokaryotes, mRNA and tRNA transcription and translation
        function and importance of polypeptide synthesis
        genes and environment affect phenotypic expression
        structure and function of proteins
    """: "MOD 5 IQ 3",
    """
        variations of genotype of offspring by modelling meiosis,
        crossing over of homologous chromosomes, fertilisation and mutations
        new combinations of genotypes produced during meiosis
        examples of autosomal, sex-linkage, co-dominance, incomplete dominance and multiple alleles
        data from pedigrees and Punnett squares
        characteristics in a population, examining frequency data, analysing single nucleotide polymorphism
    """: "MOD 5 IQ 4",
    """
        inheritance patterns in a population using, DNA sequencing and profiling
        population genetics data in conservation management, studies in inheritance of a disease or disorder, 
        relating to human evolution
    """: "MOD 5 IQ 5"
}
questions = [
    """
    Explain the phenotypic ratios of the F2 generation in both the plant and chicken
    breeding experiments. Include Punnett squares and a key to support your answer.
    """,
    """
    Sickle cell anaemia is a genetic disorder. In a family, the parents are both known
    to be heterozygous for the mutation that causes sickle cell anaemia. The couple has
    two unaffected children and is now expecting a third child. They have had an allele
    screening test to determine whether the child will have sickle cell anaemia.
    A part of the DNA profile is shown. It shows the alleles present.
    Mother Father Child 1 Child 2 Child 3
    Use the DNA profile provided to justify whether Child 3 will have sickle cell anaemia.
    """,
    """
    Zebra populations are suffering from a reduction in their gene pools due to habitat
    destruction and increasing isolation. This has led to an increase in the number of
    offspring born with coat patterns different to that of their parents. An example is
    shown.
    Explain possible reasons for the increase in these offspring.
    """
]

for question in questions:
    result = classifier(question, list(syllabus_points.keys()))
    print(f"Question: {question.strip()}")
    print("Top 2 Matches:")
    for i in range(2):
        print(f"  {i + 1}. {syllabus_points[result['labels'][i]]} with confidence {result['scores'][i]:.2f}")
    print()
