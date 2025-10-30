package ma.emsi.kharraz;


import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;

import java.util.HashMap;
import java.util.Map;

public class Test2 {

    public static void main(String[] args) {
        String response = translateText("Comment rejoindre votre club de tennis? ");
        System.out.println(response);
    }

    public static String translateText(String textToTranslate) {
        String cle = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-2.5-pro")
                .build();

        PromptTemplate promptTemplate = PromptTemplate.from("Traduis le texte suivant en anglais : {{text}}");
        Map<String, Object> variables = new HashMap<>();
        variables.put("text", textToTranslate);
        Prompt prompt = promptTemplate.apply(variables);

        return model.chat(prompt.text());
    }
}

