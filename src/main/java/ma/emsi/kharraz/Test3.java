package ma.emsi.kharraz;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.CosineSimilarity;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

public class Test3 {

    private static final String GEMINI_KEY_ENV = "GEMINI_KEY";

    public static void main(String[] args) {
        String apiKey = System.getenv(GEMINI_KEY_ENV);
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("La clé GEMINI_KEY est introuvable dans les variables d'environnement.");
            return;
        }

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .taskType(GoogleAiEmbeddingModel.TaskType.SEMANTIC_SIMILARITY)
                .outputDimensionality(300)
                .timeout(Duration.ofSeconds(60))
                .build();

        List<String[]> sentencePairs = Arrays.asList(
                new String[]{"Les chats aiment dormir au soleil.", "Les félins apprécient se prélasser dans la lumière."},
                new String[]{"Je vais acheter du pain à la boulangerie.", "Je suis en retard pour la réunion."},
                new String[]{"Le ciel est bleu aujourd'hui.", "Aujourd'hui, le ciel paraît dégagé et azur."},
                new String[]{"Il pleut beaucoup dans cette région.", "Cette région est très ensoleillée."}
        );

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        System.out.println("=== Calcul de similarité entre paires de phrases ===\n");

        for (int i = 0; i < sentencePairs.size(); i++) {
            String[] pair = sentencePairs.get(i);

            TextSegment segment1 = TextSegment.from(pair[0]);
            TextSegment segment2 = TextSegment.from(pair[1]);

            Embedding embedding1 = embeddingModel.embed(segment1).content();
            Embedding embedding2 = embeddingModel.embed(segment2).content();

            embeddingStore.add(embedding1, segment1);
            embeddingStore.add(embedding2, segment2);

            double similarity = CosineSimilarity.between(embedding1, embedding2);

            System.out.printf("Paire %d:%n", i + 1);
            System.out.printf("\tPhrase A: %s%n", pair[0]);
            System.out.printf("\tPhrase B: %s%n", pair[1]);
            System.out.printf("\tSimilarité cosinus: %.4f%n%n", similarity);
        }

        System.out.println("=== Recherche de phrases similaires ===\n");

        List<String> queries = Arrays.asList(
                "Les chats aiment la chaleur du soleil",
                "Il faut que je me dépêche pour la réunion",
                "Le temps est vraiment couvert aujourd'hui"
        );

        for (String query : queries) {
            Embedding queryEmbedding = embeddingModel.embed(query).content();

            // Utilisation de la nouvelle API avec EmbeddingSearchRequest
            EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(queryEmbedding)
                    .maxResults(2)
                    .minScore(0.0)
                    .build();

            EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
            List<EmbeddingMatch<TextSegment>> matchResults = searchResult.matches();

            System.out.printf("Requête: %s%n", query);
            for (EmbeddingMatch<TextSegment> match : matchResults) {
                System.out.printf("\tTexte: %s%n", match.embedded().text());
                System.out.printf("\tScore de similarité: %.4f%n", match.score());
            }
            System.out.println();
        }
    }
}