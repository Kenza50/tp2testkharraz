package ma.emsi.kharraz;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

/**
 * Le RAG facile !
 */
public class Test4 {

    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {
        String llmKey = System.getenv("GEMINI_KEY");

        // Modèle de chat
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .build();

        // ⭐ AJOUT : Modèle d'embedding (obligatoire pour le RAG)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(llmKey)
                .modelName("text-embedding-004")
                .build();

        // Chargement du document
        String nomDocument = "C:\\Users\\ProBook\\Documents\\Projets IntelliJ IDEA\\AI\\TPs Casablanca\\tp2testkharraz\\src\\main\\java\\infos.txt";
        Document document = FileSystemDocumentLoader.loadDocument(nomDocument);

        // Base vectorielle en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // ⭐ CORRECTION : Passer l'embeddingModel à l'ingestor
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(document);

        // Création de l'assistant avec le même embeddingModel
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)  // ⭐ Même modèle pour la recherche
                        .build())
                .build();

        // Test
        String question = "Quel est la capital de la France ?";
        String reponse = assistant.chat(question);
        System.out.println(reponse);
    }
}