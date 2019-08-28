"""
Telegram bot to send training notifications of neural network.
"""

# ReplyKeyboardMarkup:- This object represents a custom keyboard with reply options
# ReplyKeyboardRemove:- Upon receiving a message with this object, Telegram clients will remove the current custom keyboard 
# and display the default letter-keyboard. By default, custom keyboards are displayed until a new keyboard is sent by a bot. 
from telegram import  ReplyKeyboardMarkup, ReplyKeyboardRemove

# Updater:- Its purpose is to receive the updates from Telegram and to deliver them to said dispatcher.
# CommandHandler:- Handler class to handle Telegram commands. Commands are Telegram messages that start with /, 
# optionally followed by an @ and the bot’s name and/or some additional text. 
# Filters:- When using MessageHandler it is sometimes useful to have more than one filter which filters the incoming text.
from telegram.ext import Updater, CommandHandler, Filters, RegexHandler, ConversationHandler

import  logging
import numpy as np 
from io import BytesIO

class bot(object):
	"""This class interacts with telegram bot.

	It currently has following commands:

	/start: Get all command options and activate automatic notifications of command updates.
	/help: get all command options.
	/status: latest epoch result.
	/quiet: stop training update notification.
	/stoptraning: kill traning process. 

	# Arguments:
		token: String, token generated to use API
		user_id: Int, Specifying a telegram user id will filter all incoming
			commands to allow access only to a specific user. Optional, though highly recommended.
	"""

	# Constructor
	def __init__(self, token, user_id=None):

		assert isinstance(token, str), 'Token must be string'
		assert user_id is None or isinstance(user_id, int), 'User_id must be int'

		self.token = token
		self.user_id = user_id
		self.filters = None 
		self.chat_id = None # Fetched during /start
		self.bot_active = False
		self.status_message = "No status msg has been set yet"
		self.verbose = True # Automatic per epoch updates
		self.stop_train_flag = False 
		self.updater = None 

		# Enable logging
		logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
		self.logger = logging.getLogger(__name__)

		self.startup_message = "Hi, I am Netify, your neural network training notifier." \
							" send /start to activate automatic updates every epoch\n" \
							" send /help to see all options.\n" \
							" Send /status to get the latest results.\n" \
							" Send /quiet to stop getting automatic updates each epoch\n" 

	def activate_bot(self):
		"""Initiate telegram bot"""

		# Set Updater
		self.updater = Updater(self.token)
		# Register handlers with dispactchers
		dp = self.updater.dispatcher 
		# log errors
		dp.add_error_handler(self.error)

		self.filters = Filters.user(user_id=self.user_id) if self.user_id else None 
		dp.add_handler(CommandHandler("start", self.start, filters=self.filters))  # /start
		dp.add_handler(CommandHandler("help", self.help, filters=self.filters))  # /help
		dp.add_handler(CommandHandler("status", self.status, filters=self.filters))  # /get status
		dp.add_handler(CommandHandler("quiet", self.quiet, filters=self.filters))  # /stop automatic updates

		# Start bot
		self.updater.start_polling()
		self.bot_active = True 

	# Start the bot
	def start(self, bot, update):
		self.verbose = True
		update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
		self.chat_id = update.message.chat_id 

	# Stop the bot
	def stop_bot(self):
		self.updater.stop()
		self.bot_active = False 
    
	def help(self, bot, update):
		""" Telegram bot callback for the /help command. Replies the startup message"""
		update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())
		self.chat_id = update.message.chat_id

	# Stop updates
	def quiet(self, bot, update):
		self.verbose = False
		update.message.reply_text(" Automatic epoch updates turned off. Send /start to turn epoch updates back on.")

	def error(self, update, error):
		"""Log Errors caused by Updates."""
		self.logger.warning('Update "%s" caused error "%s"', update, error)
		
	def send_message(self,txt):
		assert isinstance(txt, str), 'Message text must be of type string'
		if self.verbose:
			if self.chat_id is not None:
				self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
			else:
				print('Send message failed, user did not send /start')

	def set_status(self, txt):
		assert isinstance(txt, str), 'Status Message must be of type string'
		self.status_message = txt

	def status(self, bot, update):
		update.message.reply_text(self.status_message)	