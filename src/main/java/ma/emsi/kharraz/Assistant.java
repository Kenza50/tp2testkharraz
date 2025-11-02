package ma.emsi.kharraz;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface Assistant {

    @SystemMessage("""
        Vous êtes un assistant pédagogique expert qui répond aux questions 
        basées sur le support de cours fourni. Répondez de manière claire 
        et précise. Si vous ne trouvez pas l'information dans le contexte, 
        dites-le clairement.
        """)
    String chat(@UserMessage String userMessage);
}
