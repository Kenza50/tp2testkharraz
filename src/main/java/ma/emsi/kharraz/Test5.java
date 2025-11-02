package ma.emsi.kharraz;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;


public class Test5 {

    public static void main(String[] args) {
        System.out.println("=== Démarrage du système RAG avec PDF ===\n");

        // 1. Charger le fichier PDF
        String pdfFileName = "src/main/java/ml.pdf";
        Path documentPath = Paths.get(pdfFileName);

        System.out.println("Chargement du document : " + pdfFileName);
        DocumentParser documentParser = new ApachePdfBoxDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, documentParser);

        System.out.println("Document chargé avec succès !");

        // 2. Découper le document en segments
        System.out.println("Découpage du document en segments...");
        DocumentSplitter splitter = DocumentSplitters.recursive(
                300,  // Taille maximale d'un segment (en tokens)
                20    // Chevauchement entre segments
        );
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Document découpé en " + segments.size() + " segments.\n");

        // 3. Créer le modèle d'embedding (local, gratuit)
        System.out.println("Initialisation du modèle d'embedding...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 4. Créer le store d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 5. Ingérer les segments dans le store
        System.out.println("Création des embeddings et stockage...");
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .documentSplitter(splitter)
                .build();

        // In version 1.5.0, ingest accepts Document objects
        ingestor.ingest(document);
        System.out.println("Embeddings créés et stockés avec succès !\n");

        // 6. Créer le content retriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)  // Nombre de segments pertinents à récupérer
                .minScore(0.6)  // Score minimum de similarité
                .build();

        // 7. Configurer le modèle de chat (Gemini)
        String apiKey = System.getenv("GEMINI_KEY"); // ou mettre votre clé directement

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash")
                .temperature(0.7)
                .build();

        // 8. Créer l'assistant avec mémoire de conversation
        System.out.println("Création de l'assistant conversationnel...");
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("Assistant prêt !\n");
        System.out.println("=".repeat(60));
        System.out.println("Vous pouvez maintenant poser vos questions.");
        System.out.println("Tapez 'quit' ou 'exit' pour quitter.");
        System.out.println("=".repeat(60));

        // 9. Boucle de conversation interactive
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("\n💬 Vous : ");
            String userInput = scanner.nextLine().trim();

            if (userInput.equalsIgnoreCase("quit") ||
                    userInput.equalsIgnoreCase("exit")) {
                System.out.println("\n👋 Au revoir !");
                break;
            }

            if (userInput.isEmpty()) {
                continue;
            }

            try {
                System.out.println("\n🤖 Assistant : ");
                String response = assistant.chat(userInput);
                System.out.println(response);
            } catch (Exception e) {
                System.err.println("\n❌ Erreur : " + e.getMessage());
                e.printStackTrace();
            }
        }

        scanner.close();
    }
}